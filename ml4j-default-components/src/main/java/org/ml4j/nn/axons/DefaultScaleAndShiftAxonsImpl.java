package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DefaultScaleAndShiftAxonsImpl<N extends Neurons> implements ScaleAndShiftAxons<N> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private MatrixFactory matrixFactory;

	private Matrix scaleColumnVector;
	private Matrix shiftColumnVector;

	public DefaultScaleAndShiftAxonsImpl(MatrixFactory matrixFactory, N leftNeurons, N rightNeurons) {
		super();
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.scaleColumnVector = matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), 1);
		this.shiftColumnVector = matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(), 1);
	}

	private N leftNeurons;
	private N rightNeurons;

	@Override
	public N getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public N getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		// TODO - currently mock implementation
		if (leftNeuronsActivation instanceof ImageNeuronsActivation) {
			return new AxonsActivationImpl(this, null, () -> leftNeuronsActivation,
					new ImageNeuronsActivationImpl(
							matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(),
									leftNeuronsActivation.getExampleCount()),
							(Neurons3D) getRightNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET,
							false),
					leftNeurons, rightNeurons);
		} else {
			return new AxonsActivationImpl(this, null, () -> leftNeuronsActivation,
					new NeuronsActivationImpl(
							matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(),
									leftNeuronsActivation.getExampleCount()),
							NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET),
					leftNeurons, rightNeurons);
		}
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		// TODO - currently mock implementation
		if (rightNeuronsActivation instanceof ImageNeuronsActivation) {
			return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation,
					new ImageNeuronsActivationImpl(
							matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(),
									rightNeuronsActivation.getExampleCount()),
							(Neurons3D) getLeftNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET,
							false),
					leftNeurons, rightNeurons);
		} else {
			return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation,
					new NeuronsActivationImpl(
							matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(),
									rightNeuronsActivation.getExampleCount()),
							NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET),
					leftNeurons, rightNeurons);
		}
	}

	@Override
	public ScaleAndShiftAxons<N> dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return true;
	}

	@Override
	public Matrix getScaleColumnVector() {
		return scaleColumnVector;
	}

	@Override
	public Matrix getShiftColumnVector() {
		return shiftColumnVector;
	}

	@Override
	public void adjustAxonWeights(AxonWeightsAdjustment adjustments,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		// No-op
	}

	@Override
	public AxonWeights getDetachedAxonWeights() {
		throw new UnsupportedOperationException();
	}

}
