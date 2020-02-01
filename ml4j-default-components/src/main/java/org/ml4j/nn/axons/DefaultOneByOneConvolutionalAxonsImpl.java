package org.ml4j.nn.axons;

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

public class DefaultOneByOneConvolutionalAxonsImpl implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;
	private Axons3DConfig config;
	private FullyConnectedAxons fullyConnectedAxons;

	public DefaultOneByOneConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
			FullyConnectedAxons fullyConnectedAxons, Axons3DConfig config) {
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
		this.fullyConnectedAxons = fullyConnectedAxons;
		if (!isEligible(rightNeurons, rightNeurons, config)) {
			throw new IllegalArgumentException("DefaultOneByOneConvolutionalAxonsImpl cannot be used for this configuration");
		}
	}

	public DefaultOneByOneConvolutionalAxonsImpl(AxonsFactory axonsFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
			Axons3DConfig config, Matrix connectionWeights, Matrix biases) {
		super();
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();
		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());
		if (!isEligible(rightNeurons, rightNeurons, config)) {
			throw new IllegalArgumentException("DefaultOneByOneConvolutionalAxonsImpl cannot be used for this configuration");
		}
		this.fullyConnectedAxons = axonsFactory.createFullyConnectedAxons(
				new Neurons(filterWidth * filterHeight * leftNeurons.getDepth(), leftNeurons.hasBiasUnit()),
				new Neurons(rightNeurons.getDepth(), rightNeurons.hasBiasUnit()), connectionWeights, biases);
	}

	@Override
	public void adjustAxonWeights(AxonWeightsAdjustment adjustment,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		fullyConnectedAxons.adjustAxonWeights(adjustment, adjustmentDirection);
	}

	@Override
	public Neurons3D getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons3D getRightNeurons() {
		return rightNeurons;
	}

	public Matrix reformatLeftToRightInputOneByOne(MatrixFactory matrixFactory, NeuronsActivation activations) {
		EditableMatrix out = activations.getActivations(matrixFactory).dup().asEditableMatrix();
		out.reshape(leftNeurons.getDepth(),
				leftNeurons.getWidth() * leftNeurons.getHeight() * activations.getExampleCount());
		return out;
	}

	public NeuronsActivation reformatLeftToRightInput(MatrixFactory matrixFactory,
			NeuronsActivation leftNeuronsActivation) {
		final NeuronsActivation reformatted;
			reformatted = new NeuronsActivationImpl(new Neurons(leftNeurons.getDepth(), leftNeurons.hasBiasUnit()),
					reformatLeftToRightInputOneByOne(matrixFactory, leftNeuronsActivation),
					leftNeuronsActivation.getFormat());
	
		return reformatted;

	}
	
	public static boolean isEligible(Neurons3D leftNeurons, Neurons3D rightNeurons, Axons3DConfig axonsConfig) {
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + axonsConfig.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + axonsConfig.getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (axonsConfig.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (axonsConfig.getStrideHeight());
	
		return filterWidth == 1 && filterHeight == 1 && axonsConfig.getPaddingWidth() == 0 && axonsConfig.getPaddingHeight() == 0
				&& axonsConfig.getStrideHeight() == 1 && axonsConfig.getStrideWidth() == 1;
	}

	public NeuronsActivation reformatLeftToRightOutput(MatrixFactory matrixFactory, NeuronsActivation output,
			int exampleCount) {
		output.reshape(rightNeurons.getNeuronCountExcludingBias(), exampleCount);
		return output.asImageNeuronsActivation(getRightNeurons());

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

		return new AxonsActivationImpl(this, nestedActivation.getDropoutMask(), () -> reformatted, output, leftNeurons,
				rightNeurons);
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

		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, reformattedOutput, leftNeurons,
				rightNeurons);
	}

	private NeuronsActivation reformatRightToLeftOutput(MatrixFactory matrixFactory, NeuronsActivation output,
			int exampleCount) {
		output.reshape(leftNeurons.getNeuronCountExcludingBias(), exampleCount);
		return output.asImageNeuronsActivation(leftNeurons);

	}

	public NeuronsActivation reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation input) {

		if (input.isImmutable()) {
			throw new UnsupportedOperationException();
		} else {
			input.reshape(rightNeurons.getDepth(),
					rightNeurons.getWidth() * rightNeurons.getHeight() * input.getExampleCount());
			return input;
		}
	}

	@Override
	public ConvolutionalAxons dup() {
		// TODO config.dup()
		return new DefaultOneByOneConvolutionalAxonsImpl(leftNeurons, rightNeurons, fullyConnectedAxons, config);
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
}
