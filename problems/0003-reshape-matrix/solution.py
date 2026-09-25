import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	if new_shape[0]*new_shape[1]!=len(a)*len(a[0]):
		return []
	reshaped_matrix=np.array(a).reshape(new_shape)
	return reshaped_matrix