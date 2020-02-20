package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;

public class DefaultConvolutionalAxonsImpl implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Axons3DConfig config;
	private FullyConnectedAxons fullyConnectedAxons;

	public DefaultConvolutionalAxonsImpl(AxonsFactory axonsFactory,
			Axons3DConfig config, WeightsMatrix connectionWeights,  BiasVector biases) {
		this.config = config;
		this.fullyConnectedAxons = axonsFactory.createFullyConnectedAxons(new AxonsConfig<>(
				new Neurons(config.getFilterWidth() * config.getFilterHeight() * config.getLeftNeurons().getDepth(), config.getLeftNeurons().hasBiasUnit()),
				new Neurons(config.getRightNeurons().getDepth(), config.getRightNeurons().hasBiasUnit())), 
				connectionWeights,
				biases);
	}
	
	private DefaultConvolutionalAxonsImpl(FullyConnectedAxons fullyConnectedAxons, Axons3DConfig config) {
		this.config = config;
		this.fullyConnectedAxons = fullyConnectedAxons;
	}

	@Override
	public void adjustAxonWeights(AxonWeightsAdjustment adjustment,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		fullyConnectedAxons.adjustAxonWeights(adjustment, adjustmentDirection);
	}

	@Override
	public Neurons3D getLeftNeurons() {
		return config.getLeftNeurons();
	}

	@Override
	public Neurons3D getRightNeurons() {
		return config.getRightNeurons();
	}

	public NeuronsActivation reformatLeftToRightInput(MatrixFactory matrixFactory,
			NeuronsActivation leftNeuronsActivation) {

		final NeuronsActivation reformatted;
			ImageNeuronsActivation imageAct = leftNeuronsActivation.asImageNeuronsActivation(getLeftNeurons(), DimensionScope.INPUT);
			reformatted = new NeuronsActivationImpl(
					new Neurons(getLeftNeurons().getDepth() * config.getFilterWidth() * config.getFilterHeight(), getLeftNeurons().hasBiasUnit()),
					imageAct.im2ColConv(matrixFactory, config),
					ImageNeuronsActivationFormat.ML4J_IM_TO_COL_CONV_FORMAT);
			if (!imageAct.isImmutable()) {
				imageAct.close();
			}
	
		return reformatted;

	}
	
	public NeuronsActivation reformatLeftToRightOutput(MatrixFactory matrixFactory, NeuronsActivation output,
			int exampleCount) {
		output.reshape(getRightNeurons().getNeuronCountExcludingBias(), exampleCount);
		
		return output.asImageNeuronsActivation(getRightNeurons(), DimensionScope.OUTPUT);

	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		
		if (axonsContext.getLeftHandInputDropoutKeepProbability() != 1) {
			throw new IllegalArgumentException("Convolutional layers do not support dropout");
		}

		int exampleCount = leftNeuronsActivation.getExampleCount();

		NeuronsActivation reformatted = reformatLeftToRightInput(axonsContext.getMatrixFactory(),
				leftNeuronsActivation);
		reformatted.setImmutable(true);

		AxonsActivation nestedActivation = fullyConnectedAxons.pushLeftToRight(reformatted,
				previousRightToLeftActivation, axonsContext);
		
		NeuronsActivation output = reformatLeftToRightOutput(axonsContext.getMatrixFactory(),
				nestedActivation.getPostDropoutOutput(), exampleCount);

		if (!axonsContext.isTrainingContext()) {
			reformatted.close();
		} else {
			reformatted.setImmutable(true);
		}

		if (!leftNeuronsActivation.isImmutable()) {
			leftNeuronsActivation.close();
		}

		return new AxonsActivationImpl(this, nestedActivation.getDropoutMask(), () -> reformatted, output);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		
		if (axonsContext.getLeftHandInputDropoutKeepProbability() != 1) {
			throw new IllegalArgumentException("Convolutional layers do not support dropout");
		}

		int exampleCount = previousLeftToRightActivation.getPostDropoutOutput().getExampleCount();

		NeuronsActivation reformattedInput = reformatRightToLeftInput(axonsContext.getMatrixFactory(),
				rightNeuronsActivation);

		AxonsActivation axonsActivation = fullyConnectedAxons.pushRightToLeft(reformattedInput,
				previousLeftToRightActivation, axonsContext);

		NeuronsActivation reformattedOutput = reformatRightToLeftOutput(axonsContext.getMatrixFactory(),
				axonsActivation.getPostDropoutOutput(), exampleCount);

		axonsActivation.getPostDropoutOutput().close();

		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, reformattedOutput);
	}

	private NeuronsActivation reformatRightToLeftOutput(MatrixFactory matrixFactory, NeuronsActivation output,
			int exampleCount) {

		float[] data = new float[getLeftNeurons().getDepth() * getLeftNeurons().getWidth()
				* getLeftNeurons().getHeight() * exampleCount];

		Images images = new MultiChannelImages(data, getLeftNeurons().getDepth(), getLeftNeurons().getHeight(),
				getLeftNeurons().getWidth(), config.getPaddingHeight(), config.getPaddingWidth(), exampleCount);

		images.im2colConvImport(matrixFactory, output.getActivations(matrixFactory), config.getFilterHeight(), config.getFilterWidth(),
				config.getStrideHeight(), config.getStrideWidth());

		return new ImageNeuronsActivationImpl(getLeftNeurons(), images,
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);

	}

	public NeuronsActivation reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation input) {

		if (input.isImmutable()) {
			throw new UnsupportedOperationException();
		} else {
			input.reshape(getRightNeurons().getDepth(),
					getRightNeurons().getWidth() * getRightNeurons().getHeight() * input.getExampleCount());
			return input;
		}
	}

	@Override
	public ConvolutionalAxons dup() {
		return new DefaultConvolutionalAxonsImpl(fullyConnectedAxons.dup(), config.dup());
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return !axonsContext.isWithFreezeOut();
	}

	@Override
	public AxonWeights getDetachedAxonWeights() {
		return fullyConnectedAxons.getDetachedAxonWeights();
	}

	@Override
	public Axons3DConfig getConfig() {
		return config;
	}

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return fullyConnectedAxons.optimisedFor();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return fullyConnectedAxons.isSupported(format) 
				&& NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET
				.equals(format.getFeatureOrientation());
	}
	
	@Override
	public AxonsType getAxonsType() {
		return AxonsType.getBaseType(AxonsBaseType.CONVOLUTIONAL);
	}
}
