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
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.ml4j.nn.neurons.format.features.DimensionScope;
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

	public DefaultAveragePoolingAxonsImpl(MatrixFactory matrixFactory,
			Axons3DConfig config) {
		this.matrixFactory = matrixFactory;
		this.config = config;
	}


	@Override
	public Neurons3D getLeftNeurons() {
		return config.getLeftNeurons();
	}

	@Override
	public Neurons3D getRightNeurons() {
		return config.getRightNeurons();
	}

	public Matrix reformatLeftToRightInput(MatrixFactory matrixFactory, NeuronsActivation leftNeuronsActivation) {

		ImageNeuronsActivation imageNeuronsActivation = leftNeuronsActivation.asImageNeuronsActivation(getLeftNeurons(),
				DimensionScope.INPUT);

		Matrix reformatted = imageNeuronsActivation.im2ColPool(matrixFactory, config);

		if (imageNeuronsActivation != leftNeuronsActivation) {
			imageNeuronsActivation.close();
		}

		return reformatted;

	}

	public ImageNeuronsActivation reformatLeftToRightOutput(MatrixFactory matrixFactory, NeuronsActivation output,
			int exampleCount) {
		output.reshape(getRightNeurons().getNeuronCountExcludingBias(), exampleCount);
		return output.asImageNeuronsActivation(getRightNeurons(), DimensionScope.OUTPUT);

	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

	
		ImageNeuronsActivation inputImageActivation = leftNeuronsActivation.asImageNeuronsActivation(getLeftNeurons(), DimensionScope.INPUT);

		int inputMatrixRows = inputImageActivation.getRows();
		int inputMatrixColumns = inputImageActivation.getColumns();
		int exampleCount = inputImageActivation.getExampleCount();

		ImageNeuronsActivation onesActivation = new ImageNeuronsActivationImpl(
				axonsContext.getMatrixFactory().createOnes(inputMatrixRows, inputMatrixColumns),
				inputImageActivation.getNeurons(), inputImageActivation.getFormat(), false);

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

	
		// Obtain pooled feature averages
		Matrix preFormattedOutput = 
				reformatted.columnSums().asEditableMatrix().diviRowVector(counts);
		
		reformatted.close();

		// Reformat back to output shape
		preFormattedOutput.asEditableMatrix().reshape(getRightNeurons().getNeuronCountExcludingBias(), exampleCount);
		
		ImageNeuronsActivation output = new ImageNeuronsActivationImpl(preFormattedOutput, getRightNeurons(),
				ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);
				
		// Close the pooled feature averages activations
		preFormattedOutput.close();

		// If the leftNeuronsActivation wassn't immutable, close it.
		if (!leftNeuronsActivation.isImmutable()) {
			leftNeuronsActivation.close();
		}

		return new AxonsActivationImpl(this, null, () -> null, output);

	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		int filterElementCount = config.getFilterWidth() * config.getFilterWidth();

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
					return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, reformattedOutput);
				}

			}

		}
	}

	private NeuronsActivation reformatRightToLeftOutput(MatrixFactory matrixFactory,
			NeuronsActivationFeatureOrientation featureOrientation, Matrix output, int exampleCount) {

		float[] data = new float[getLeftNeurons().getDepth() * getLeftNeurons().getWidth()
				* getLeftNeurons().getHeight() * exampleCount];

		Images images = new MultiChannelImages(data, getLeftNeurons().getDepth(), getLeftNeurons().getHeight(),
				getLeftNeurons().getWidth(), config.getPaddingHeight(), config.getPaddingWidth(), exampleCount);

		images.im2colPoolImport(matrixFactory, output, config.getFilterHeight(), config.getFilterWidth(), config.getStrideHeight(),
				config.getStrideWidth());

		return new ImageNeuronsActivationImpl(getLeftNeurons(), images, ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT, false);
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

	@Override
	public AxonsType getAxonsType() {
		return AxonsType.getBaseType(AxonsBaseType.AVERAGE_POOLING);
	}

}
