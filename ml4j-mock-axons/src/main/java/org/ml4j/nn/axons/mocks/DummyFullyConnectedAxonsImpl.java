package org.ml4j.nn.axons.mocks;

import java.util.Optional;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonWeights;
import org.ml4j.nn.axons.AxonWeightsAdjustment;
import org.ml4j.nn.axons.AxonWeightsAdjustmentDirection;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsActivationImpl;
import org.ml4j.nn.axons.AxonsBaseType;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsType;
import org.ml4j.nn.axons.FullyConnectedAxons;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;

public class DummyFullyConnectedAxonsImpl implements FullyConnectedAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private MatrixFactory matrixFactory;

	public DummyFullyConnectedAxonsImpl(MatrixFactory matrixFactory, Neurons leftNeurons, Neurons rightNeurons) {
		super();
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	private Neurons leftNeurons;
	private Neurons rightNeurons;

	@Override
	public Neurons getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		int exampleCount = leftNeuronsActivation.getExampleCount();

		if (!axonsContext.isTrainingContext() && !leftNeuronsActivation.isImmutable()) {
			leftNeuronsActivation.close();
		}

		return new AxonsActivationImpl(this, null, () -> leftNeuronsActivation,
				new NeuronsActivationImpl(getRightNeurons(),
						matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(), exampleCount),
						ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT));
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation,
				new NeuronsActivationImpl(getLeftNeurons(),
						matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(),
								rightNeuronsActivation.getExampleCount()),
						ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT));
	}

	@Override
	public FullyConnectedAxons dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return true;
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
		return AxonsType.getBaseType(AxonsBaseType.FULLY_CONNECTED);
	}

}
