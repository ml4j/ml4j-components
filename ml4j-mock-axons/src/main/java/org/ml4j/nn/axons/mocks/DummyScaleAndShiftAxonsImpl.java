package org.ml4j.nn.axons.mocks;

import java.util.Optional;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonWeights;
import org.ml4j.nn.axons.AxonWeightsAdjustment;
import org.ml4j.nn.axons.AxonWeightsAdjustmentDirection;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsActivationImpl;
import org.ml4j.nn.axons.AxonsBaseType;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsType;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

public class DummyScaleAndShiftAxonsImpl<N extends Neurons> implements ScaleAndShiftAxons<N> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private MatrixFactory matrixFactory;

	private Matrix scaleColumnVector;
	private Matrix shiftColumnVector;

	public DummyScaleAndShiftAxonsImpl(MatrixFactory matrixFactory, N leftNeurons, N rightNeurons) {
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

		int exampleCount = leftNeuronsActivation.getExampleCount();

		if (!axonsContext.isTrainingContext()) {
			leftNeuronsActivation.close();
		}

		return new AxonsActivationImpl(this, null, () -> leftNeuronsActivation,
				new NeuronsActivationImpl(getRightNeurons(),
						matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(), exampleCount),
						NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET));
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation,
				new NeuronsActivationImpl(getLeftNeurons(),
						matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(),
								rightNeuronsActivation.getExampleCount()),
						NeuronsActivationFormat.ROWS_SPAN_FEATURE_SET));
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

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}
	
	@Override
	public AxonsType getAxonsType() {
		return AxonsType.getBaseType(AxonsBaseType.SCALE_AND_SHIFT);
	}

}
