package org.ml4j.nn.axons;

import java.util.Arrays;
import java.util.Optional;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.Dimension;
import org.ml4j.nn.neurons.format.features.DimensionScope;
import org.ml4j.nn.neurons.format.features.FeaturesFormatImpl;

public class DefaultOneByOneConvolutionalAxonsImpl implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Axons3DConfig config;
	private FullyConnectedAxons fullyConnectedAxons;

	private DefaultOneByOneConvolutionalAxonsImpl(
			FullyConnectedAxons fullyConnectedAxons, Axons3DConfig config) {
		this.config = config;
		this.fullyConnectedAxons = fullyConnectedAxons;
		if (!isEligible(config)) {
			throw new IllegalArgumentException("DefaultOneByOneConvolutionalAxonsImpl cannot be used for this configuration");
		}
	}

	public DefaultOneByOneConvolutionalAxonsImpl(AxonsFactory axonsFactory,
			Axons3DConfig config, WeightsMatrix connectionWeights, BiasVector biases) {
		this.config = config;
		if (!isEligible(config)) {
			throw new IllegalArgumentException("DefaultOneByOneConvolutionalAxonsImpl cannot be used for this configuration");
		}
		this.fullyConnectedAxons = axonsFactory.createFullyConnectedAxons(new AxonsConfig<>(
				new Neurons(config.getFilterWidth() * config.getFilterHeight() * config.getLeftNeurons().getDepth(), config.getLeftNeurons().hasBiasUnit()),
				new Neurons(config.getRightNeurons().getDepth(), config.getRightNeurons().hasBiasUnit())), 
				connectionWeights,
				biases);
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

	public Matrix reformatLeftToRightInputOneByOne(MatrixFactory matrixFactory, NeuronsActivation activations) {
		EditableMatrix out = activations.getActivations(matrixFactory).dup().asEditableMatrix();
		out.reshape(config.getLeftNeurons().getDepth(),
				config.getLeftNeurons().getWidth() * config.getLeftNeurons().getHeight() * activations.getExampleCount());
		return out;
	}

	public NeuronsActivation reformatLeftToRightInput(MatrixFactory matrixFactory,
			NeuronsActivation leftNeuronsActivation)  {
		final NeuronsActivation reformatted;
			reformatted = new NeuronsActivationImpl(new Neurons(config.getLeftNeurons().getDepth(), config.getLeftNeurons().hasBiasUnit()),
					reformatLeftToRightInputOneByOne(matrixFactory, leftNeuronsActivation),
					// TODO
					new NeuronsActivationFormat<>(leftNeuronsActivation.getFeatureOrientation(),
							new FeaturesFormatImpl(Arrays.asList(Dimension.DEPTH)),
									Arrays.asList(Dimension.HEIGHT, Dimension.WIDTH, Dimension.EXAMPLE)));	
		return reformatted;

	}
	
	public static boolean isEligible(Axons3DConfig axonsConfig) {
		return axonsConfig.getFilterWidth() == 1 && axonsConfig.getFilterHeight() == 1 && axonsConfig.getPaddingWidth() == 0 && axonsConfig.getPaddingHeight() == 0
				&& axonsConfig.getStrideHeight() == 1 && axonsConfig.getStrideWidth() == 1;
	}

	public NeuronsActivation reformatLeftToRightOutput(MatrixFactory matrixFactory, NeuronsActivation output,
			int exampleCount) {
		output.reshape(config.getRightNeurons().getNeuronCountExcludingBias(), exampleCount);
		return output.asImageNeuronsActivation(getRightNeurons(), DimensionScope.OUTPUT);

	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

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
		output.reshape(config.getLeftNeurons().getNeuronCountExcludingBias(), exampleCount);
		return output.asImageNeuronsActivation(config.getLeftNeurons(), DimensionScope.OUTPUT);

	}

	public NeuronsActivation reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation input) {

		if (input.isImmutable()) {
			throw new UnsupportedOperationException();
		} else {
			input.reshape(config.getRightNeurons().getDepth(),
					config.getRightNeurons().getWidth() * config.getRightNeurons().getHeight() * input.getExampleCount());
			return input;
		}
	}

	@Override
	public ConvolutionalAxons dup() {
		return new DefaultOneByOneConvolutionalAxonsImpl(fullyConnectedAxons.dup(), config.dup());
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
