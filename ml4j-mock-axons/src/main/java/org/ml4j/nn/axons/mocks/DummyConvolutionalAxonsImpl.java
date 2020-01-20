package org.ml4j.nn.axons.mocks;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.AxonWeights;
import org.ml4j.nn.axons.AxonWeightsAdjustment;
import org.ml4j.nn.axons.AxonWeightsAdjustmentDirection;
import org.ml4j.nn.axons.Axons3DConfig;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsActivationImpl;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.ConvolutionalAxons;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

public class DummyConvolutionalAxonsImpl implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private MatrixFactory matrixFactory;
	private Axons3DConfig config;

	public DummyConvolutionalAxonsImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config) {
		super();
		this.matrixFactory = matrixFactory;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
	}

	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;

	@Override
	public void adjustAxonWeights(AxonWeightsAdjustment adjustment,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		// No-op
	}

	@Override
	public Neurons3D getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons3D getRightNeurons() {
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
				new ImageNeuronsActivationImpl(
						matrixFactory.createMatrix(rightNeurons.getNeuronCountExcludingBias(), exampleCount),
						getRightNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false),
				leftNeurons, rightNeurons);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation,
				new ImageNeuronsActivationImpl(
						matrixFactory.createMatrix(leftNeurons.getNeuronCountExcludingBias(),
								rightNeuronsActivation.getExampleCount()),
						getLeftNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false),
				leftNeurons, rightNeurons);
	}

	@Override
	public ConvolutionalAxons dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return false;
	}

	@Override
	public AxonWeights getDetachedAxonWeights() {
		throw new UnsupportedOperationException();
	}

	@Override
	public Axons3DConfig getConfig() {
		return config;
	}

	@Override
	public Optional<NeuronsActivationFeatureOrientation> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public List<NeuronsActivationFeatureOrientation> supports() {
		return Arrays.asList(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
	}
}
