package org.ml4j.nn.axons;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DefaultConvolutionalAxonsImpl implements ConvolutionalAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Neurons3D leftNeurons;
	private Neurons3D rightNeurons;
	private Axons3DConfig config;
	private FullyConnectedAxons fullyConnectedAxons;

	public DefaultConvolutionalAxonsImpl(Neurons3D leftNeurons, Neurons3D rightNeurons,
			FullyConnectedAxons fullyConnectedAxons, Axons3DConfig config) {
		super();
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
		this.config = config;
		this.fullyConnectedAxons = fullyConnectedAxons;
	}

	public DefaultConvolutionalAxonsImpl(AxonsFactory axonsFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
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
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());

		final NeuronsActivation reformatted;
		if (isEligableOneByOne(filterWidth, filterHeight)) {
			reformatted = new NeuronsActivationImpl(
					reformatLeftToRightInputOneByOne(matrixFactory, leftNeuronsActivation),
					NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		} else {
			reformatted = new NeuronsActivationImpl(
					leftNeuronsActivation.asImageNeuronsActivation(leftNeurons).im2ColConv(matrixFactory, filterHeight,
							filterWidth, config.getStrideHeight(), config.getStrideWidth(), config.getPaddingHeight(),
							config.getPaddingWidth()),
					NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		}
		return reformatted;

	}

	private boolean isEligableOneByOne(int filterWidth, int filterHeight) {
		return filterWidth == 1 && filterHeight == 1 && config.getPaddingWidth() == 0 && config.getPaddingHeight() == 0
				&& config.getStrideHeight() == 1 && config.getStrideWidth() == 1;

	}

	public Matrix reformatLeftToRightOutput(MatrixFactory matrixFactory, NeuronsActivation output, int exampleCount) {
		Matrix reformattedOutput = output.getActivations(matrixFactory);
		reformattedOutput.asEditableMatrix().reshape(rightNeurons.getNeuronCountExcludingBias(), exampleCount);

		return reformattedOutput;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		NeuronsActivation reformatted = reformatLeftToRightInput(axonsContext.getMatrixFactory(),
				leftNeuronsActivation);

		AxonsActivation nestedActivation = fullyConnectedAxons.pushLeftToRight(reformatted, previousRightToLeftActivation, axonsContext);

		Matrix output = reformatLeftToRightOutput(axonsContext.getMatrixFactory(),
				nestedActivation.getPostDropoutOutput(), leftNeuronsActivation.getExampleCount());

		return new AxonsActivationImpl(this, nestedActivation.getDropoutMask(), () -> reformatted, new ImageNeuronsActivationImpl(output,
				getRightNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false), leftNeurons,
				rightNeurons);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		NeuronsActivation reformattedInput = reformatRightToLeftInput(axonsContext.getMatrixFactory(),
				rightNeuronsActivation);

		AxonsActivation axonsActivation = fullyConnectedAxons.pushRightToLeft(reformattedInput,
				previousLeftToRightActivation, axonsContext);

		NeuronsActivation reformattedOutput = reformatRightToLeftOutput(axonsContext.getMatrixFactory(),
				axonsActivation.getPostDropoutOutput().getActivations(axonsContext.getMatrixFactory()),
				previousLeftToRightActivation.getPostDropoutOutput().getExampleCount());

		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, reformattedOutput, leftNeurons,
				rightNeurons);
	}

	private NeuronsActivation reformatRightToLeftOutput(MatrixFactory matrixFactory, Matrix output, int exampleCount) {

		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;

		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());

		if (isEligableOneByOne(filterWidth, filterHeight)) {
			EditableMatrix out = output.dup().asEditableMatrix();
			out.reshape(leftNeurons.getNeuronCountExcludingBias(), exampleCount);
			return new ImageNeuronsActivationImpl(out, leftNeurons, NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);
		} else {

			float[] data = new float[getLeftNeurons().getDepth() * getLeftNeurons().getWidth()
					* getLeftNeurons().getHeight() * exampleCount];

			Images images = new MultiChannelImages(data, getLeftNeurons().getDepth(), getLeftNeurons().getHeight(),
					getLeftNeurons().getWidth(), config.getPaddingHeight(), config.getPaddingWidth(), exampleCount);

			images.im2colConvImport(matrixFactory, output, filterHeight, filterWidth, config.getStrideHeight(),
					config.getStrideWidth());

			return new ImageNeuronsActivationImpl(getLeftNeurons(), images, false);

		}
	}

	public NeuronsActivation reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation input) {
		EditableMatrix m = input.getActivations(matrixFactory).asEditableMatrix();
		m.reshape(rightNeurons.getDepth(),
				rightNeurons.getWidth() * rightNeurons.getHeight() * input.getExampleCount());
		return new NeuronsActivationImpl(m, input.getFeatureOrientation());
	}

	@Override
	public ConvolutionalAxons dup() {
		// TODO config.dup()
		return new DefaultConvolutionalAxonsImpl(leftNeurons, rightNeurons, fullyConnectedAxons, config);
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

}
