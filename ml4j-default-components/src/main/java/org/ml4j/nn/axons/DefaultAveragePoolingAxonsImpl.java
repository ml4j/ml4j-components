package org.ml4j.nn.axons;

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
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultAveragePoolingAxonsImpl implements AveragePoolingAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultAveragePoolingAxonsImpl.class);

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

		ImageNeuronsActivation imageNeuronsActivation = leftNeuronsActivation.asImageNeuronsActivation(leftNeurons);
		
		NeuronsActivation reformatted = new NeuronsActivationImpl(
				imageNeuronsActivation.im2ColPool(matrixFactory, filterHeight,
						filterWidth, config.getStrideHeight(), config.getStrideWidth(), config.getPaddingHeight(),
						config.getPaddingWidth()),
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);

		if (imageNeuronsActivation != leftNeuronsActivation) {
			imageNeuronsActivation.close();
		}
		
		return reformatted;

	}
	
	public NeuronsActivation reformatLeftToRightOutput(MatrixFactory matrixFactory, NeuronsActivation output, int exampleCount) {
		if (output instanceof ImageNeuronsActivation) {
			Matrix reformattedOutput = output.getActivations(matrixFactory);
			reformattedOutput.asEditableMatrix().reshape(rightNeurons.getNeuronCountExcludingBias(), exampleCount);
			output.close();
			return new ImageNeuronsActivationImpl(reformattedOutput,
					getRightNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);
			
		} else {
			Matrix reformattedOutput = output.getActivations(matrixFactory);
			reformattedOutput.asEditableMatrix().reshape(rightNeurons.getNeuronCountExcludingBias(), exampleCount);
			return output.asImageNeuronsActivation(getRightNeurons());
			//return new ImageNeuronsActivationImpl(reformattedOutput,
			//		getRightNeurons(), NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);
		}

	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		NeuronsActivation inputMatrix = leftNeuronsActivation;
		
		int inputMatrixRows = inputMatrix.getRows();
		int inputMatrixColumns = inputMatrix.getColumns();

		if (!(inputMatrix instanceof ImageNeuronsActivation)) {
			inputMatrix = inputMatrix.asImageNeuronsActivation(leftNeurons);
		}
		ImageNeuronsActivation c = (ImageNeuronsActivation)inputMatrix;
	
		Matrix inputOnes = axonsContext.getMatrixFactory().createOnes(inputMatrixRows, inputMatrixColumns);
		
		NeuronsActivation reformattedActivation = reformatLeftToRightInput(axonsContext.getMatrixFactory(),
				leftNeuronsActivation);
		
		Matrix reformatted = reformattedActivation.getActivations(matrixFactory);
		
		ImageNeuronsActivation onesAct = new ImageNeuronsActivationImpl(inputOnes, (Neurons3D)c.getNeurons(), 
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET, false);
		
		
		NeuronsActivation countsAct = reformatLeftToRightInput(axonsContext.getMatrixFactory(), 
				onesAct);
		
		onesAct.close();
		
		Matrix counts = countsAct.getActivations(matrixFactory);
		
		EditableMatrix counts2 = counts.columnSums().asEditableMatrix();
		for (int i = 0; i < counts2.getLength(); i++) {
			 if (counts2.get(i) == 0) {
				 counts2.put(i, 1);
			 }
		}
				
		Matrix origOutput = reformatted.columnSums().asEditableMatrix().diviRowVector(counts2);
		countsAct.close();

		NeuronsActivation preFormattedOutput = new NeuronsActivationImpl(origOutput,
				NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET);


		NeuronsActivation output = reformatLeftToRightOutput(axonsContext.getMatrixFactory(), preFormattedOutput,
				leftNeuronsActivation.getExampleCount());
		
		//if (!axonsContext.isTrainingContext()) {
			reformattedActivation.close();
		//}
		
		if (!leftNeuronsActivation.isImmutable()) {
			//TODO
			leftNeuronsActivation.close();
		}

		return new AxonsActivationImpl(
				this, null, () -> reformattedActivation,output,
				leftNeurons, rightNeurons);

	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		NeuronsActivation reformattedInput = reformatRightToLeftInput(axonsContext.getMatrixFactory(),
				rightNeuronsActivation);

		Matrix reformatted = reformattedInput.getActivations(axonsContext.getMatrixFactory());

		int filterWidth = leftNeurons.getWidth() + (2 * config.getPaddingWidth())
				+ (1 - rightNeurons.getWidth()) * (config.getStrideWidth());
		int filterHeight = leftNeurons.getHeight() + (2 * config.getPaddingHeight())
				+ (1 - rightNeurons.getHeight()) * (config.getStrideHeight());
		int filterElementCount = filterWidth * filterHeight;

		try (InterrimMatrix reformattedRow = reformatRightToLeftInput(axonsContext.getMatrixFactory(),
				rightNeuronsActivation).getActivations(axonsContext.getMatrixFactory()).asInterrimMatrix()) {

			try (InterrimMatrix reformattedAvgRow = reformattedRow.div(filterElementCount).asInterrimMatrix()) {
				try (InterrimMatrix reformattedAvgMatrix = axonsContext.getMatrixFactory()
						.createMatrix(filterElementCount, reformattedAvgRow.getColumns() * reformattedAvgRow.getRows())
						.asInterrimMatrix()) {
					for (int r = 0; r < reformattedAvgMatrix.getRows(); r++) {
						reformattedAvgMatrix.asEditableMatrix().putRow(r, reformattedAvgRow);
					}
					LOGGER.debug("Output average pooling axons:" + reformattedAvgMatrix.getRows() + ":"
							+ reformattedAvgMatrix.getColumns());

					LOGGER.debug("Reformatted average pooling axons:" + reformatted.getRows() + ":"
							+ reformatted.getColumns());

					Matrix preFormattedOutput = reformattedAvgMatrix;

					NeuronsActivation reformattedOutput = reformatRightToLeftOutput(axonsContext.getMatrixFactory(),
							preFormattedOutput, previousLeftToRightActivation.getPostDropoutOutput().getExampleCount());

					if (!rightNeuronsActivation.isImmutable()) {
						rightNeuronsActivation.close();
					}
					return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, reformattedOutput,
							leftNeurons, rightNeurons);
				}

			}

		}
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

		float[] data = new float[getLeftNeurons().getDepth() * getLeftNeurons().getWidth()
				* getLeftNeurons().getHeight() * exampleCount];

		Images images = new MultiChannelImages(data, getLeftNeurons().getDepth(), getLeftNeurons().getHeight(),
				getLeftNeurons().getWidth(), config.getPaddingHeight(), config.getPaddingWidth(), exampleCount);

		images.im2colPoolImport(matrixFactory, output, filterHeight, filterWidth, config.getStrideHeight(),
				config.getStrideWidth());

		return new ImageNeuronsActivationImpl(getLeftNeurons(), images, false);
	}

	public NeuronsActivation reformatRightToLeftInput(MatrixFactory matrixFactory, NeuronsActivation input) {
		return input;
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

}
