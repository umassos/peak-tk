
from imblearn.over_sampling import ADASYN 
from imblearn.under_sampling import RandomUnderSampler

# oversampling
def oversampling(X, y)
	ada = ADASYN() 
	X_resampled_ADA, y_resampled_ADA = ada.fit_resample(X, y) 
	return (X_resampled_ADA, y_resampled_ADA)

# undersampling
def undersampling(X, y, random_seed=0)
	rus = RandomUnderSampler(random_state=random_seed)
	X_resampled_RUS, y_resampled_RUS = rus.fit_resample(X, y)
	return (X_resampled_RUS, y_resampled_RUS)

