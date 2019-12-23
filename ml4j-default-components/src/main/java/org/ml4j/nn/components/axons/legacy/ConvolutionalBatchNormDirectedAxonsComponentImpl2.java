package org.ml4j.nn.components.axons.legacy;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.neurons.Neurons3D;

public class ConvolutionalBatchNormDirectedAxonsComponentImpl2 extends BatchNormDirectedAxonsComponentImpl<Neurons3D> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	public ConvolutionalBatchNormDirectedAxonsComponentImpl2(MatrixFactory matrixFactory, ScaleAndShiftAxons<Neurons3D> scaleAndShiftAxons,
			Matrix exponentiallyWeightedAverageInputChannelMeans,
			Matrix exponentiallyWeightedAverageInputChannelVariances) {
		super(scaleAndShiftAxons, expandChannelValuesToFeatureValues(matrixFactory, scaleAndShiftAxons.getRightNeurons(), exponentiallyWeightedAverageInputChannelMeans),
				expandChannelValuesToFeatureValues(matrixFactory, scaleAndShiftAxons.getRightNeurons(), exponentiallyWeightedAverageInputChannelVariances));
	}

	public ConvolutionalBatchNormDirectedAxonsComponentImpl2(ScaleAndShiftAxons<Neurons3D> scaleAndShiftAxons) {
		super(scaleAndShiftAxons);
	}

	public ConvolutionalBatchNormDirectedAxonsComponentImpl2(ScaleAndShiftAxons<Neurons3D> scaleAndShiftAxons,
			Matrix exponentiallyWeightedAverageInputFeatureMeans,
			Matrix exponentiallyWeightedAverageInputFeatureVariances) {
		super(scaleAndShiftAxons, exponentiallyWeightedAverageInputFeatureMeans,
				exponentiallyWeightedAverageInputFeatureVariances);
	}

	@Override
	protected Matrix getVarianceColumnVector(Matrix matrix, MatrixFactory matrixFactory, Matrix meanColumnVector, boolean isTraining) {
		
		if (isTraining) {
			if (matrix.getColumns() == 1) {
				if (exponentiallyWeightedAverageInputFeatureVariances != null) {
					return exponentiallyWeightedAverageInputFeatureVariances;
				} else {
					throw new IllegalStateException("Unable to calcuate mean and variance for batch "
							+ "norm on a single example - no exponentially weighted average available");
				}
			}
		} else {
			if (exponentiallyWeightedAverageInputFeatureVariances != null) {
				return exponentiallyWeightedAverageInputFeatureVariances;
			} else {
				throw new IllegalStateException("Unable to obtain variance for batch "
						+ "norm when not in training - rolling variance not supplied");
			}
		}
		
		int examples = matrix.getColumns();
		matrix.asEditableMatrix().reshape(getAxons().getRightNeurons().getDepth(), 
				getAxons().getRightNeurons().getWidth() * getAxons().getRightNeurons().getHeight() * examples);
		Matrix channelVariances =  super.getVarianceColumnVector(matrix, matrixFactory, meanColumnVector, isTraining);
		matrix.asEditableMatrix().reshape(getAxons().getRightNeurons().getNeuronCountExcludingBias(), examples);
		return expandChannelValuesToFeatureValues(matrixFactory, axons.getRightNeurons(), channelVariances);
	}
	
	public static Matrix expandChannelValuesToFeatureValues(MatrixFactory matrixFactory, Neurons3D rightNeurons, Matrix channelValues) {
		if (channelValues == null) return null;
		float[] channelValuesArray = channelValues.getRowByRowArray();
		float[] channelValuesExpanded = new float[rightNeurons.getNeuronCountExcludingBias()];
		int index = 0;
		for (int channel = 0; channel < channelValuesArray.length; channel++) {
			for (int i = 0;  i < rightNeurons.getWidth() * rightNeurons.getHeight(); i++) {
				channelValuesExpanded[index++] = channelValuesArray[channel];
			}
		}
		return matrixFactory.createMatrixFromRowsByRowsArray(channelValuesExpanded.length, 1, channelValuesExpanded);

	}

	@Override
	protected Matrix getMeanColumnVector(Matrix matrix, MatrixFactory matrixFactory, boolean isTraining) {
		
		if (isTraining) {
			if (matrix.getColumns() == 1) {
				if (exponentiallyWeightedAverageInputFeatureMeans != null) {
					return exponentiallyWeightedAverageInputFeatureMeans;
				} else {
					return matrix;
				}
			}
		} else {
			if (exponentiallyWeightedAverageInputFeatureMeans != null) {
				return exponentiallyWeightedAverageInputFeatureMeans;
			} else {
				throw new IllegalStateException("Unable to obtain mean for batch "
						+ "norm when not in training - rolling mean not supplied");
			}
		}
		
		int examples = matrix.getColumns();
		matrix.asEditableMatrix().reshape(getAxons().getRightNeurons().getDepth(), 
				getAxons().getRightNeurons().getWidth() * getAxons().getRightNeurons().getHeight() * examples);
		Matrix channelVariances =  super.getMeanColumnVector(matrix, matrixFactory, isTraining);
		matrix.asEditableMatrix().reshape(getAxons().getRightNeurons().getNeuronCountExcludingBias(), examples);
		return expandChannelValuesToFeatureValues(matrixFactory, axons.getRightNeurons(), channelVariances);
	}
	
	@Override
	public ConvolutionalBatchNormDirectedAxonsComponentImpl2 dup() {
		return new ConvolutionalBatchNormDirectedAxonsComponentImpl2(axons.dup(), exponentiallyWeightedAverageInputFeatureMeans.dup(),
				exponentiallyWeightedAverageInputFeatureVariances.dup());
	}

}
