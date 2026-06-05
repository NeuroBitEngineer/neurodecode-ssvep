from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import binomtest
from sklearn.cross_decomposition import CCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
RESULTS_FOLDER = PROJECT_FOLDER / 'results'
RESULTS_FOLDER.mkdir(exist_ok=True)

os.environ['MNE_DATA'] = str(PROJECT_FOLDER / 'mne_data')
os.environ['MNE_CONFIG'] = str(PROJECT_FOLDER / 'mne_config')
os.environ['HOME'] = str(PROJECT_FOLDER)

from moabb.datasets import Wang2016
from moabb.paradigms import SSVEP

SUBJECT_LIST = [1, 2, 3, 4, 5, 6]
SELECTED_TARGET_FREQUENCIES = np.array([8.0, 10.0, 12.0, 15.0])
WINDOW_DURATIONS = [1.2, 2.0, 4.0, 6.0]
CHANNEL_SETS = {'occipital': ['O1', 'Oz', 'O2'],
                'parieto_occipital': ['PO3', 'POz', 'PO4', 'O1', 'Oz', 'O2']}
FILTER_BANKS = [(7, 14), (14, 22), (22, 32), (32, 45)]
SAMPLING_RATE = 250
CHANCE_ACCURACY = 0.25
PERMUTATION_COUNT = 5000


def make_reference_signals(frequency, number_of_samples, sampling_rate=SAMPLING_RATE, number_of_harmonics=3):
    
    time_axis = np.arange(number_of_samples) / sampling_rate
    reference_components = []

    for harmonic in range(1, number_of_harmonics + 1):
        reference_components.append(np.sin(2 * np.pi * harmonic * frequency * time_axis))
        reference_components.append(np.cos(2 * np.pi * harmonic * frequency * time_axis))

    return np.array(reference_components).T


def predict_with_cca(eeg_trials, target_frequencies, sampling_rate=SAMPLING_RATE, number_of_harmonics=3):
    
    predicted_classes = []

    for trial_data in eeg_trials:
        trial_matrix = trial_data.T
        class_scores = []

        for frequency in target_frequencies:
            reference_matrix = make_reference_signals(frequency, trial_data.shape[-1], sampling_rate, number_of_harmonics)
            cca_model = CCA(n_components=1, max_iter=1000)
            transformed_eeg, transformed_reference = cca_model.fit_transform(trial_matrix, reference_matrix)
            correlation = np.corrcoef(transformed_eeg[:, 0], transformed_reference[:, 0])[0, 1]
            class_scores.append(abs(correlation))

        predicted_classes.append(int(np.argmax(class_scores)))

    return np.array(predicted_classes)


def predict_with_target_power(eeg_trials, target_frequencies, sampling_rate=SAMPLING_RATE):
    
    predicted_classes = []

    for trial_data in eeg_trials:
        class_scores = []

        for target_frequency in target_frequencies:
            channel_scores = []

            for channel_values in trial_data:
                frequency_axis, power_density = welch(channel_values, fs=sampling_rate, nperseg=min(256, channel_values.shape[-1]))
                target_index = np.argmin(np.abs(frequency_axis - target_frequency))
                channel_scores.append(power_density[target_index])

            class_scores.append(float(np.mean(channel_scores)))

        predicted_classes.append(int(np.argmax(class_scores)))

    return np.array(predicted_classes)


def filter_eeg_trials(eeg_trials, low_frequency, high_frequency, sampling_rate=SAMPLING_RATE):
    
    sos_filter = butter(4, [low_frequency, high_frequency], btype='bandpass', fs=sampling_rate, output='sos')

    return sosfiltfilt(sos_filter, eeg_trials, axis=-1)


def predict_with_filter_bank_cca(eeg_trials, target_frequencies):
    
    predicted_classes = []
    filter_weights = np.array([(filter_index + 1) ** -1.25 + 0.25 for filter_index in range(len(FILTER_BANKS))])

    filtered_trial_sets = [filter_eeg_trials(eeg_trials, low_frequency, high_frequency)
                           for low_frequency, high_frequency in FILTER_BANKS]

    for trial_index in range(eeg_trials.shape[0]):
        weighted_scores = np.zeros(len(target_frequencies))

        for filter_index, filtered_trials in enumerate(filtered_trial_sets):
            trial_predictions = []
            trial_data = filtered_trials[trial_index]
            trial_matrix = trial_data.T

            for frequency in target_frequencies:
                reference_matrix = make_reference_signals(frequency, trial_data.shape[-1])
                cca_model = CCA(n_components=1, max_iter=1000)
                transformed_eeg, transformed_reference = cca_model.fit_transform(trial_matrix, reference_matrix)
                correlation = np.corrcoef(transformed_eeg[:, 0], transformed_reference[:, 0])[0, 1]
                trial_predictions.append(abs(correlation))

            weighted_scores += filter_weights[filter_index] * np.array(trial_predictions)

        predicted_classes.append(int(np.argmax(weighted_scores)))

    return np.array(predicted_classes)


def solve_trca_filter(class_trials):
    
    number_of_trials, number_of_channels, _ = class_trials.shape
    cross_trial_covariance = np.zeros((number_of_channels, number_of_channels))
    within_trial_covariance = np.zeros((number_of_channels, number_of_channels))

    demeaned_trials = class_trials - class_trials.mean(axis=-1, keepdims=True)

    for first_trial_index in range(number_of_trials):
        first_trial = demeaned_trials[first_trial_index]
        within_trial_covariance += first_trial @ first_trial.T

        for second_trial_index in range(first_trial_index + 1, number_of_trials):
            second_trial = demeaned_trials[second_trial_index]
            cross_trial_covariance += first_trial @ second_trial.T + second_trial @ first_trial.T

    regularized_covariance = within_trial_covariance + 1e-6 * np.eye(number_of_channels)
    eigenvalues, eigenvectors = eigh(cross_trial_covariance, regularized_covariance)

    return eigenvectors[:, np.argmax(eigenvalues)]


def fit_trca_templates(eeg_trials, class_labels):
    
    model_by_class = {}

    for class_label in sorted(np.unique(class_labels)):
        class_trials = eeg_trials[class_labels == class_label]
        spatial_filter = solve_trca_filter(class_trials)
        class_template = class_trials.mean(axis=0)
        model_by_class[int(class_label)] = {'spatial_filter': spatial_filter,
                                            'class_template': class_template}

    return model_by_class


def predict_with_trca(eeg_trials, model_by_class):
    
    predicted_classes = []

    for trial_data in eeg_trials:
        class_scores = []

        for class_label, class_model in model_by_class.items():
            spatial_filter = class_model['spatial_filter']
            class_template = class_model['class_template']
            projected_trial = spatial_filter @ trial_data
            projected_template = spatial_filter @ class_template
            correlation = np.corrcoef(projected_trial, projected_template)[0, 1]
            class_scores.append((class_label, abs(correlation)))

        predicted_classes.append(max(class_scores, key=lambda item: item[1])[0])

    return np.array(predicted_classes)


def cross_validate_trca(eeg_trials, class_labels):
    
    split_count = min(4, np.min(np.bincount(class_labels)))
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=138)
    predicted_classes = np.zeros_like(class_labels)

    for train_indices, test_indices in splitter.split(eeg_trials, class_labels):
        trca_model = fit_trca_templates(eeg_trials[train_indices], class_labels[train_indices])
        predicted_classes[test_indices] = predict_with_trca(eeg_trials[test_indices], trca_model)

    return predicted_classes


def cross_validate_filter_bank_trca(eeg_trials, class_labels):
    
    split_count = min(4, np.min(np.bincount(class_labels)))
    splitter = StratifiedKFold(n_splits=split_count, shuffle=True, random_state=138)
    predicted_classes = np.zeros_like(class_labels)
    filter_weights = np.array([(filter_index + 1) ** -1.25 + 0.25 for filter_index in range(len(FILTER_BANKS))])
    filtered_trial_sets = [filter_eeg_trials(eeg_trials, low_frequency, high_frequency)
                           for low_frequency, high_frequency in FILTER_BANKS]

    for train_indices, test_indices in splitter.split(eeg_trials, class_labels):
        model_sets = [fit_trca_templates(filtered_trials[train_indices], class_labels[train_indices])
                      for filtered_trials in filtered_trial_sets]

        for test_index in test_indices:
            weighted_scores = {}

            for filter_index, trca_model in enumerate(model_sets):
                trial_data = filtered_trial_sets[filter_index][test_index]

                for class_label, class_model in trca_model.items():
                    spatial_filter = class_model['spatial_filter']
                    class_template = class_model['class_template']
                    projected_trial = spatial_filter @ trial_data
                    projected_template = spatial_filter @ class_template
                    correlation = abs(np.corrcoef(projected_trial, projected_template)[0, 1])
                    weighted_scores[class_label] = weighted_scores.get(class_label, 0.0) + filter_weights[filter_index] * correlation

            predicted_classes[test_index] = max(weighted_scores.items(), key=lambda item: item[1])[0]

    return predicted_classes


def make_statistics(true_classes, predicted_classes, random_seed=138):
    
    correct_count = int((true_classes == predicted_classes).sum())
    trial_count = int(len(true_classes))
    accuracy = accuracy_score(true_classes, predicted_classes)
    binomial_p_value = binomtest(correct_count, trial_count, CHANCE_ACCURACY, alternative='greater').pvalue
    random_generator = np.random.default_rng(random_seed)
    permutation_accuracies = []

    for _ in range(PERMUTATION_COUNT):
        permuted_classes = random_generator.permutation(true_classes)
        permutation_accuracies.append((permuted_classes == predicted_classes).mean())

    permutation_accuracies = np.array(permutation_accuracies)
    permutation_p_value = (np.sum(permutation_accuracies >= accuracy) + 1) / (PERMUTATION_COUNT + 1)

    return correct_count, trial_count, accuracy, binomial_p_value, permutation_p_value


def load_selected_wang_trials(subject_list, selected_channels, window_duration):
    
    wang_dataset = Wang2016(subjects=subject_list)
    ssvep_paradigm = SSVEP(fmin=7, fmax=45, tmin=0.0, tmax=window_duration, channels=selected_channels)
    public_eeg_trials, public_labels, public_metadata = ssvep_paradigm.get_data(dataset=wang_dataset, subjects=subject_list)
    label_frequencies = pd.Series(public_labels).astype(float)
    selected_trial_rows = label_frequencies.isin(SELECTED_TARGET_FREQUENCIES)
    selected_public_eeg_trials = public_eeg_trials[selected_trial_rows]
    selected_public_labels = label_frequencies[selected_trial_rows].to_numpy()
    frequency_to_class = {frequency: class_index for class_index, frequency in enumerate(SELECTED_TARGET_FREQUENCIES)}
    selected_public_classes = np.array([frequency_to_class[frequency] for frequency in selected_public_labels])

    return selected_public_eeg_trials, selected_public_classes, public_metadata[selected_trial_rows]


def run_expansion():
    
    result_rows = []

    for window_duration in WINDOW_DURATIONS:
        for channel_set_name, selected_channels in CHANNEL_SETS.items():
            eeg_trials, class_labels, metadata = load_selected_wang_trials(SUBJECT_LIST, selected_channels, window_duration)
            model_predictions = {'target_power': predict_with_target_power(eeg_trials, SELECTED_TARGET_FREQUENCIES),
                                 'cca': predict_with_cca(eeg_trials, SELECTED_TARGET_FREQUENCIES),
                                 'filter_bank_cca': predict_with_filter_bank_cca(eeg_trials, SELECTED_TARGET_FREQUENCIES),
                                 'trca_cross_validated': cross_validate_trca(eeg_trials, class_labels),
                                 'filter_bank_trca_cross_validated': cross_validate_filter_bank_trca(eeg_trials, class_labels)}

            for model_name, predicted_classes in model_predictions.items():
                correct_count, trial_count, accuracy, binomial_p_value, permutation_p_value = make_statistics(class_labels, predicted_classes)
                result_rows.append({'subjects': ','.join(str(subject) for subject in SUBJECT_LIST),
                                    'window_duration_seconds': window_duration,
                                    'channel_set': channel_set_name,
                                    'channels': len(selected_channels),
                                    'trials': trial_count,
                                    'samples': eeg_trials.shape[-1],
                                    'model': model_name,
                                    'accuracy': accuracy,
                                    'correct': correct_count,
                                    'binomial_p_value': binomial_p_value,
                                    'permutation_p_value': permutation_p_value})

    results = pd.DataFrame(result_rows)
    results.to_csv(RESULTS_FOLDER / 'wang2016_expanded_model_comparison.csv', index=False)
    print(results.sort_values(['window_duration_seconds', 'channel_set', 'accuracy'],
                              ascending=[True, True, False]).to_string(index=False))


if __name__ == '__main__':
    run_expansion()
