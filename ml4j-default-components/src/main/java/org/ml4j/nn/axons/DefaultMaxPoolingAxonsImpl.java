package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Images;
import org.ml4j.images.MultiChannelImages;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultMaxPoolingAxonsImpl implements MaxPoolingAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultMaxPoolingAxonsImpl.class);

	private MatrixFactory matrixFactory;
	private Axons3DConfig config;
	private boolean scaleOutputs;

	public DefaultMaxPoolingAxonsImpl(MatrixFactory matrixFactory,
			Axons3DConfig config, boolean scaleOutputs) {
		this.matrixFactory = matrixFactory;
		this.config = config;
		this.scaleOutputs = scaleOutputs;
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
	
		ImageNeuronsActivation act = leftNeuronsActivation.asImageNeuronsActivation(getLeftNeurons(), DimensionScope.INPUT);

		NeuronsActivation reformatted = new NeuronsActivationImpl(
				new Neurons(config.getFilterWidth() * config.getFilterHeight(), getLeftNeurons().hasBiasUnit()),
				act.im2ColPool(matrixFactory, config),
				ImageNeuronsActivationFormat.ML4J_IM_TO_COL_POOL_FORMAT);
		if (!act.isImmutable()) {
			act.close();
		}

		if (scaleOutputs) {
			int outputDim = (int) (this.getRightNeurons().getNeuronCountIncludingBias()
					/ this.getRightNeurons().getDepth());
			int inputDim = (int) (this.getLeftNeurons().getNeuronCountIncludingBias() / getLeftNeurons().getDepth());

			float scaleDown = inputDim / outputDim;
			reformatted.applyValueModifier(v -> v * scaleDown);
		}

		return reformatted;

	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		int exampleCount = leftNeuronsActivation.getExampleCount();

		NeuronsActivation reformattedActivation = reformatLeftToRightInput(axonsContext.getMatrixFactory(),
				leftNeuronsActivation);

		Matrix reformatted = reformattedActivation.getActivations(matrixFactory);

		EditableMatrix maxes = (EditableMatrix) axonsContext.getMatrixFactory().createMatrix(reformatted.getRows(),
				reformatted.getColumns());

		EditableMatrix origOutput = (EditableMatrix) axonsContext.getMatrixFactory().createMatrix(1,
				reformatted.getColumns());
		int[] maxInts = reformatted.columnArgmaxs();
		for (int c = 0; c < reformatted.getColumns(); c++) {
			if (maxInts[c] != -1) {
				maxes.put(maxInts[c], c, 1);
				origOutput.put(0, c, reformatted.get(maxInts[c], c));
			}
		}
	
		AxonsDropoutMask maxesDropoutMask = new AxonsDropoutMaskImpl(maxes, AxonsDropoutMaskType.INPUT);
		
		origOutput.reshape(getRightNeurons().getNeuronCountExcludingBias(), exampleCount);
		
		ImageNeuronsActivation output = new ImageNeuronsActivationImpl(origOutput, getRightNeurons(),
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);

		if (!leftNeuronsActivation.isImmutable()) {
			leftNeuronsActivation.close();
		}
		reformattedActivation.close();

		return new AxonsActivationImpl(this, maxesDropoutMask, () -> null, output);

	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		Matrix inputActivations = rightNeuronsActivation.getActivations(axonsContext.getMatrixFactory());

		int exampleCount = previousLeftToRightActivation.getPostDropoutOutput().getExampleCount();

		LOGGER.debug("Reformatted max pooling axons:" + inputActivations.getRows() + ":" + inputActivations.getColumns());

		AxonsDropoutMask axonsDropoutMask = previousLeftToRightActivation.getDropoutMask();

		try (InterrimMatrix outputMatrix = axonsDropoutMask.getDropoutMask().asInterrimMatrix()) {

			EditableMatrix reformatted = axonsContext.getMatrixFactory()
					.createMatrix(outputMatrix.getRows(), outputMatrix.getColumns()).asEditableMatrix();
			for (int r = 0; r < reformatted.getRows(); r++) {
				reformatted.putRow(r, inputActivations);
			}
			// reformatted.close();
			outputMatrix.asEditableMatrix().muli(reformatted);

			Matrix preFormattedOutput = outputMatrix;

			NeuronsActivation reformattedOutput = reformatRightToLeftOutput(axonsContext.getMatrixFactory(),
					preFormattedOutput, rightNeuronsActivation.getFeatureOrientation(), exampleCount);

			reformatted.close();
			if (!rightNeuronsActivation.isImmutable()) {
				rightNeuronsActivation.close();
			}
			return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, reformattedOutput);
		}
	}

	private NeuronsActivation reformatRightToLeftOutput(MatrixFactory matrixFactory, Matrix output,
			NeuronsActivationFeatureOrientation featureOrientation, int exampleCount) {

		int inputWidth = getLeftNeurons().getWidth();
		int inputHeight = getLeftNeurons().getHeight();
		int outputWidth = getRightNeurons().getWidth();
		int outputHeight = getRightNeurons().getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;

		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());

		float[] data = new float[getLeftNeurons().getDepth() * getLeftNeurons().getWidth()
				* getLeftNeurons().getHeight() * exampleCount];

		Images images = new MultiChannelImages(data, getLeftNeurons().getDepth(), getLeftNeurons().getHeight(),
				getLeftNeurons().getWidth(), config.getPaddingHeight(), config.getPaddingWidth(), exampleCount);

		images.im2colPoolImport(matrixFactory, output, filterHeight, filterWidth, config.getStrideHeight(),
				config.getStrideWidth());

		return new ImageNeuronsActivationImpl(getLeftNeurons(), images, ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);
	}

	public NeuronsActivation reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation input) {
		return input;
	}

	@Override
	public MaxPoolingAxons dup() {
		throw new UnsupportedOperationException();
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return false;
	}

	@Override
	public Axons3DConfig getConfig() {
		return config;
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
		return AxonsType.getBaseType(AxonsBaseType.MAX_POOLING);
	}

}
