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
import org.ml4j.nn.neurons.NeuronsActivationFormat;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultAveragePoolingAxonsImpl implements AveragePoolingAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultAveragePoolingAxonsImpl.class);

	@SuppressWarnings("unused")
	private MatrixFactory matrixFactory;
	private Axons3DConfig config;

	public DefaultAveragePoolingAxonsImpl(MatrixFactory matrixFactory, Neurons3D leftNeurons, Neurons3D rightNeurons,
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
	public Neurons3D getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons3D getRightNeurons() {
		return rightNeurons;
	}

	public Matrix reformatLeftToRightInput(MatrixFactory matrixFactory, NeuronsActivation leftNeuronsActivation) {
		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

		int inputWidthWithPadding = inputWidth + config.getPaddingWidth() * 2;

		int inputHeightWithPadding = inputHeight + config.getPaddingHeight() * 2;
		int filterWidth = inputWidthWithPadding + (1 - outputWidth) * (config.getStrideWidth());

		int filterHeight = inputHeightWithPadding + (1 - outputHeight) * (config.getStrideHeight());

		ImageNeuronsActivation imageNeuronsActivation = leftNeuronsActivation.asImageNeuronsActivation(leftNeurons);

		Matrix reformatted = imageNeuronsActivation.im2ColPool(matrixFactory, filterHeight, filterWidth,
				config.getStrideHeight(), config.getStrideWidth(), config.getPaddingHeight(), config.getPaddingWidth());

		if (imageNeuronsActivation != leftNeuronsActivation) {
			imageNeuronsActivation.close();
		}

		return reformatted;

	}

	public ImageNeuronsActivation reformatLeftToRightOutput(MatrixFactory matrixFactory, NeuronsActivation output,
			int exampleCount) {
		output.reshape(rightNeurons.getNeuronCountExcludingBias(), exampleCount);
		return output.asImageNeuronsActivation(getRightNeurons());

	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		// Obtain the inbound activation as an image activation, and create an
		// activation of ones of the same shape.

		// TODO
		ImageNeuronsActivation inputImageActivation = leftNeuronsActivation.asImageNeuronsActivation(leftNeurons);

		int inputMatrixRows = inputImageActivation.getRows();
		int inputMatrixColumns = inputImageActivation.getColumns();
		int exampleCount = inputImageActivation.getExampleCount();

		ImageNeuronsActivation onesActivation = new ImageNeuronsActivationImpl(
				axonsContext.getMatrixFactory().createOnes(inputMatrixRows, inputMatrixColumns),
				inputImageActivation.getNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);

		// Reformat the input and the ones activations.

		Matrix reformatted = reformatLeftToRightInput(axonsContext.getMatrixFactory(), inputImageActivation);
		Matrix reformattedOnes = reformatLeftToRightInput(axonsContext.getMatrixFactory(), onesActivation);

		EditableMatrix counts = reformattedOnes.columnSums().asEditableMatrix();
		for (int i = 0; i < counts.getLength(); i++) {
			if (counts.get(i) == 0) {
				counts.put(i, 1);
			}
		}
		onesActivation.close();
		reformattedOnes.close();

		Neurons averageNeurons = new Neurons(1, false);

		// Obtain pooled feature averages
		NeuronsActivation preFormattedOutput = new NeuronsActivationImpl(averageNeurons,
				reformatted.columnSums().asEditableMatrix().diviRowVector(counts),
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);
		reformatted.close();

		// Reformat back to output shape.
		ImageNeuronsActivation output = reformatLeftToRightOutput(axonsContext.getMatrixFactory(), preFormattedOutput,
				exampleCount);

		// Close the pooled feature averages activations
		preFormattedOutput.close();

		// If the leftNeuronsActivation wassn't immutable, close it.
		if (!leftNeuronsActivation.isImmutable()) {
			leftNeuronsActivation.close();
		}

		return new AxonsActivationImpl(this, null, () -> null, output, leftNeurons, rightNeurons);

	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		// Matrix reformatted =
		// reformattedInput.getActivations(axonsContext.getMatrixFactory());

		int filterWidth = leftNeurons.getWidth() + (2 * config.getPaddingWidth())
				+ (1 - rightNeurons.getWidth()) * (config.getStrideWidth());
		int filterHeight = leftNeurons.getHeight() + (2 * config.getPaddingHeight())
				+ (1 - rightNeurons.getHeight()) * (config.getStrideHeight());
		int filterElementCount = filterWidth * filterHeight;

		try (InterrimMatrix reformattedRow = rightNeuronsActivation.getActivations(axonsContext.getMatrixFactory())
				.asInterrimMatrix()) {

			try (InterrimMatrix reformattedAvgRow = reformattedRow.div(filterElementCount).asInterrimMatrix()) {
				try (InterrimMatrix reformattedAvgMatrix = axonsContext.getMatrixFactory()
						.createMatrix(filterElementCount, reformattedAvgRow.getColumns() * reformattedAvgRow.getRows())
						.asInterrimMatrix()) {
					for (int r = 0; r < reformattedAvgMatrix.getRows(); r++) {
						reformattedAvgMatrix.asEditableMatrix().putRow(r, reformattedAvgRow);
					}
					LOGGER.debug("Output average pooling axons:" + reformattedAvgMatrix.getRows() + ":"
							+ reformattedAvgMatrix.getColumns());

					LOGGER.debug("Reformatted average pooling axons:" + rightNeuronsActivation.getRows() + ":"
							+ rightNeuronsActivation.getColumns());

					Matrix preFormattedOutput = reformattedAvgMatrix;

					NeuronsActivation reformattedOutput = reformatRightToLeftOutput(axonsContext.getMatrixFactory(),
							rightNeuronsActivation.getFeatureOrientation(), preFormattedOutput,
							previousLeftToRightActivation.getPostDropoutOutput().getExampleCount());

					if (!rightNeuronsActivation.isImmutable()) {
						rightNeuronsActivation.close();
					}
					return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, reformattedOutput,
							leftNeurons, rightNeurons);
				}

			}

		}
	}

	private NeuronsActivation reformatRightToLeftOutput(MatrixFactory matrixFactory,
			NeuronsActivationFeatureOrientation featureOrientation, Matrix output, int exampleCount) {

		int inputWidth = leftNeurons.getWidth();
		int inputHeight = leftNeurons.getHeight();
		int outputWidth = rightNeurons.getWidth();
		int outputHeight = rightNeurons.getHeight();

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

		return new ImageNeuronsActivationImpl(getLeftNeurons(), images, featureOrientation, false);
	}

	@Override
	public AveragePoolingAxons dup() {
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

}
