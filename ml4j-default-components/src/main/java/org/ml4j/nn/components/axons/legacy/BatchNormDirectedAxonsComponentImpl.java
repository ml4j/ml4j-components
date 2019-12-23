package org.ml4j.nn.components.axons.legacy;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.components.axons.BatchNormDirectedAxonsComponent;
import org.ml4j.nn.components.axons.DirectedAxonsComponentActivation;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentBase;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

/**
 * Default implementation of batch-norm DirectedSynapses
 * 
 * @author Michael Lavelle
 *
 * @param <L>
 *            The Neurons on the left hand side of these batch-norm
 *            DirectedSynapses.
 * @param <R>
 *            The Neurons on the right hand side of these batch-norm
 *            DirectedSynapses.
 */
public class BatchNormDirectedAxonsComponentImpl<N extends Neurons> extends DirectedAxonsComponentBase<N, N, ScaleAndShiftAxons<N>> implements BatchNormDirectedAxonsComponent<N, N> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	protected Matrix exponentiallyWeightedAverageInputFeatureMeans;
	protected Matrix exponentiallyWeightedAverageInputFeatureVariances;
	protected float betaForExponentiallyWeightedAverages;

	/**
	 * @param leftNeurons
	 *            The left neurons.
	 * @param rightNeurons
	 *            The right neurons.
	 * @param scaleAndShiftAxons
	 *            The scale and shift axons.
	 */
	public BatchNormDirectedAxonsComponentImpl(ScaleAndShiftAxons<N> scaleAndShiftAxons) {
		super(scaleAndShiftAxons);
		// this.betaForExponentiallyWeightedAverages = 0.9;
		// TODO ML - Make this configurable
		this.betaForExponentiallyWeightedAverages = 0.9997f;
	}
	
	private BatchNormDirectedAxonsComponentImpl(ScaleAndShiftAxons<N> scaleAndShiftAxons,
			Matrix exponentiallyWeightedAverageInputFeatureMeans,
			Matrix exponentiallyWeightedAverageInputFeatureVariances, float betaForExponentiallyWeightedAverages) {
		super(scaleAndShiftAxons);
		this.exponentiallyWeightedAverageInputFeatureMeans = exponentiallyWeightedAverageInputFeatureMeans;
		this.exponentiallyWeightedAverageInputFeatureVariances = exponentiallyWeightedAverageInputFeatureVariances;
		this.betaForExponentiallyWeightedAverages = betaForExponentiallyWeightedAverages;
	}



	/**
	 * @param leftNeurons
	 *            The left neurons.
	 * @param rightNeurons
	 *            The right neurons.
	 * @param scaleAndShiftAxons
	 *            The scale and shift axons.
	 */
	public BatchNormDirectedAxonsComponentImpl(ScaleAndShiftAxons<N> scaleAndShiftAxons, 
			Matrix exponentiallyWeightedAverageInputFeatureMeans, Matrix exponentiallyWeightedAverageInputFeatureVariances ) {
		super(scaleAndShiftAxons);
		// this.betaForExponentiallyWeightedAverages = 0.9;
		// TODO ML - Make this configurable
		this.betaForExponentiallyWeightedAverages = 0.9997f;
		this.exponentiallyWeightedAverageInputFeatureMeans =exponentiallyWeightedAverageInputFeatureMeans == null ? null : exponentiallyWeightedAverageInputFeatureMeans.dup();
		this.exponentiallyWeightedAverageInputFeatureVariances = exponentiallyWeightedAverageInputFeatureVariances == null ? null : exponentiallyWeightedAverageInputFeatureVariances.dup();

	}

	@Override
	public DirectedAxonsComponentActivation forwardPropagate(NeuronsActivation input, AxonsContext axonsContext) {

		Matrix activations = input.getActivations(axonsContext.getMatrixFactory());
		Matrix meanColumnVector = getMeanColumnVector(activations, axonsContext.getMatrixFactory(), axonsContext.isTrainingContext());

		Matrix varianceColumnVector = getVarianceColumnVector(activations, axonsContext.getMatrixFactory(),
				meanColumnVector, axonsContext.isTrainingContext());
		Matrix xhat = activations.asEditableMatrix().subiColumnVector(meanColumnVector).diviColumnVector(getStdDevColumnVector(varianceColumnVector));

		NeuronsActivation xhatN = null;
		if (input instanceof ImageNeuronsActivation) {
			xhatN = new ImageNeuronsActivationImpl(xhat, (Neurons3D)this.getAxons().getRightNeurons(), input.getFeatureOrientation(), false);
		} else {
			xhatN = new NeuronsActivationImpl(xhat, input.getFeatureOrientation());
		}
		
		AxonsActivation axonsActivation =
				axons.pushLeftToRight(xhatN, null, axonsContext);

		return new BatchNormDirectedAxonsComponentActivationImpl(this, axons, axonsActivation,
				meanColumnVector, varianceColumnVector, axonsContext);
	}
	
	private Matrix getStdDevColumnVector(Matrix varianceColumnVector) {
		EditableMatrix stdDev = varianceColumnVector.dup().asEditableMatrix();
		float epsilion = 0.001f;
		for (int i = 0; i < stdDev.getLength(); i++) {
			float variance = stdDev.get(i);
			float stdDevValue = (float)Math.sqrt(variance + epsilion);
			stdDev.put(i, stdDevValue);
		}
		return stdDev;
	}
	

	/**
	 * Naive implementation to construct a variance row vector with an entry for
	 * each feature.
	 * 
	 * @param matrix
	 *            The input matrix
	 * @param matrixFactory
	 *            The matrix factory.
	 * @param meanRowVector
	 *            The mean row vector.
	 * @return A row vector the the variances.
	 */
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
		
		EditableMatrix columnVector = (EditableMatrix)matrixFactory.createMatrix(matrix.getRows(), 1);
		for (int r = 0; r < matrix.getRows(); r++) {
			float total = 0f;
			float count = 0;
			for (int c = 0; c < matrix.getColumns(); c++) {
				float diff = (matrix.get(r, c) - meanColumnVector.get(r, 0));
				total = total + diff * diff;
				count++;
			}
			float variance = total / count;
			columnVector.put(r, 0, variance);
		}
		return columnVector;
	}

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
		
		return matrix.rowSums().asEditableMatrix().divi(matrix.getColumns());
	}

	@Override
	public float getBetaForExponentiallyWeightedAverages() {
		return betaForExponentiallyWeightedAverages;
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureMeans() {
		return exponentiallyWeightedAverageInputFeatureMeans;
	}

	@Override
	public Matrix getExponentiallyWeightedAverageInputFeatureVariances() {
		return exponentiallyWeightedAverageInputFeatureVariances;
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureMeans(Matrix exponentiallyWeightedAverageInputFeatureMeans) {
		this.exponentiallyWeightedAverageInputFeatureMeans = exponentiallyWeightedAverageInputFeatureMeans;
	}

	@Override
	public void setExponentiallyWeightedAverageInputFeatureVariances(
			Matrix exponentiallyWeightedAverageInputFeatureVariances) {
		this.exponentiallyWeightedAverageInputFeatureVariances = exponentiallyWeightedAverageInputFeatureVariances;
	}

	@Override
	public BatchNormDirectedAxonsComponent<N, N> dup() {
		return new BatchNormDirectedAxonsComponentImpl<>(axons.dup(), exponentiallyWeightedAverageInputFeatureMeans.dup(),
				exponentiallyWeightedAverageInputFeatureVariances.dup(), betaForExponentiallyWeightedAverages);
	}
}
