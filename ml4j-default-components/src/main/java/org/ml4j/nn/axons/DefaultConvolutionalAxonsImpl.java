package org.ml4j.nn.axons;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons1D;
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
				new Neurons1D(filterWidth * filterHeight * leftNeurons.getDepth(), leftNeurons.hasBiasUnit()),
				new Neurons1D(rightNeurons.getDepth(), rightNeurons.hasBiasUnit()), connectionWeights, biases);
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
			reformatted = new NeuronsActivationImpl(new Neurons1D(leftNeurons.getDepth(), leftNeurons.hasBiasUnit()),
					reformatLeftToRightInputOneByOne(matrixFactory, leftNeuronsActivation),
					NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		} else {
			ImageNeuronsActivation imageAct = leftNeuronsActivation.asImageNeuronsActivation(leftNeurons);
			reformatted = new NeuronsActivationImpl(new Neurons1D(leftNeurons.getDepth() * filterWidth * filterHeight, leftNeurons.hasBiasUnit()),
					imageAct.im2ColConv(matrixFactory, filterHeight,
							filterWidth, config.getStrideHeight(), config.getStrideWidth(), config.getPaddingHeight(),
							config.getPaddingWidth()),
					NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
			if (!imageAct.isImmutable()) {
				imageAct.close();
			}

		}
		return reformatted;

	}

	private boolean isEligableOneByOne(int filterWidth, int filterHeight) {
		return filterWidth == 1 && filterHeight == 1 && config.getPaddingWidth() == 0 && config.getPaddingHeight() == 0
				&& config.getStrideHeight() == 1 && config.getStrideWidth() == 1;

	}

	public NeuronsActivation reformatLeftToRightOutput(MatrixFactory matrixFactory, NeuronsActivation output, int exampleCount) {
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
		
		AxonsActivation nestedActivation = fullyConnectedAxons.pushLeftToRight(reformatted, previousRightToLeftActivation, axonsContext);

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
				axonsActivation.getPostDropoutOutput(),
				exampleCount);
		
		axonsActivation.getPostDropoutOutput().close();

		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, reformattedOutput, leftNeurons,
				rightNeurons);
	}

	private NeuronsActivation reformatRightToLeftOutput(MatrixFactory matrixFactory, NeuronsActivation output, int exampleCount) {

		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;

		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());

		if (isEligableOneByOne(filterWidth, filterHeight)) {
			output.reshape(leftNeurons.getNeuronCountExcludingBias(), exampleCount);
			return output.asImageNeuronsActivation(leftNeurons);
		} else {

			float[] data = new float[getLeftNeurons().getDepth() * getLeftNeurons().getWidth()
					* getLeftNeurons().getHeight() * exampleCount];

			Images images = new MultiChannelImages(data, getLeftNeurons().getDepth(), getLeftNeurons().getHeight(),
					getLeftNeurons().getWidth(), config.getPaddingHeight(), config.getPaddingWidth(), exampleCount);

			images.im2colConvImport(matrixFactory, output.getActivations(matrixFactory), filterHeight, filterWidth, config.getStrideHeight(),
					config.getStrideWidth());

			return new ImageNeuronsActivationImpl(getLeftNeurons(), images, output.getFeatureOrientation(), false);

		}
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

	@Override
	public Optional<NeuronsActivationFeatureOrientation> optimisedFor() {
		return fullyConnectedAxons.optimisedFor();
	}

	@Override
	public List<NeuronsActivationFeatureOrientation> supports() {
		return NeuronsActivationFeatureOrientation.intersectLists(fullyConnectedAxons.supports(), 
				Arrays.asList(NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET));
	}
}
