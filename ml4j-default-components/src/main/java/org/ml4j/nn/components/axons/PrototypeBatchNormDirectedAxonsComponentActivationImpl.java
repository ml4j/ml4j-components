package org.ml4j.nn.components.axons;

import java.util.List;
import java.util.Optional;
import java.util.function.Supplier;

import org.ml4j.EditableMatrix;
import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.axons.AxonsActivation;
import org.ml4j.nn.axons.AxonsContext;
import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.axons.AxonsGradientImpl;
import org.ml4j.nn.axons.ScaleAndShiftAxons;
import org.ml4j.nn.axons.TrainableAxons;
import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.axons.base.DirectedAxonsComponentActivationBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class PrototypeBatchNormDirectedAxonsComponentActivationImpl<N extends Neurons> extends
		DirectedAxonsComponentActivationBase<ScaleAndShiftAxons<N>> implements DirectedAxonsComponentActivation {

	private Matrix meanColumnVector;
	private Matrix varianceColumnVector;
	private BatchNormDirectedAxonsComponent<?, ?> batchNormDirectedAxonsComponent;

	/**
	 * @param synapses                     The synapses.
	 * @param scaleAndShiftAxons           The scale and shift axons.
	 * @param inputActivation              The input activation.
	 * @param axonsActivation              The axons activation.
	 * @param activationFunctionActivation The activation function activation.
	 * @param outputActivation             The output activation.
	 */
	public PrototypeBatchNormDirectedAxonsComponentActivationImpl(
			BatchNormDirectedAxonsComponent<N, ScaleAndShiftAxons<N>> batchNormAxonsComponent,
			ScaleAndShiftAxons<?> scaleAndShiftAxons, AxonsActivation scaleAndShiftAxonsActivation,
			Matrix meanColumnVector, Matrix varianceColumnVector, AxonsContext axonsContext) {
		super(batchNormAxonsComponent, scaleAndShiftAxonsActivation, axonsContext);
		this.meanColumnVector = meanColumnVector;
		this.varianceColumnVector = varianceColumnVector;
		this.batchNormDirectedAxonsComponent = batchNormAxonsComponent;
	}

	private Matrix getStdDevColumnVector(Matrix varianceColumnVector) {
		EditableMatrix stdDev = varianceColumnVector.dup().asEditableMatrix();
		float epsilion = 0.01f;
		for (int i = 0; i < stdDev.getLength(); i++) {
			float variance = stdDev.get(i);
			float stdDevValue = (float) Math.sqrt(variance + epsilion);
			stdDev.put(i, stdDevValue);
		}
		return stdDev;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {

		// Build up the exponentially weighted averages

		Matrix exponentiallyWeightedAverageMean = batchNormDirectedAxonsComponent
				.getExponentiallyWeightedAverageInputFeatureMeans();

		float beta = batchNormDirectedAxonsComponent.getBetaForExponentiallyWeightedAverages();

		try (InterrimMatrix varianceColumnVectorMul1MinusBeta = varianceColumnVector.mul(1 - (float) beta)
				.asInterrimMatrix()) {

			try (InterrimMatrix meanColumnVectorMul1MinusBeta = meanColumnVector.mul(1 - (float) beta)
					.asInterrimMatrix()) {

				if (exponentiallyWeightedAverageMean == null) {
					exponentiallyWeightedAverageMean = meanColumnVector.dup();
				} else {
					exponentiallyWeightedAverageMean.asEditableMatrix().muli(beta).addi(meanColumnVectorMul1MinusBeta);
				}
				batchNormDirectedAxonsComponent
						.setExponentiallyWeightedAverageInputFeatureMeans(exponentiallyWeightedAverageMean);

				Matrix exponentiallyWeightedAverageVariance = batchNormDirectedAxonsComponent
						.getExponentiallyWeightedAverageInputFeatureVariances();

				if (exponentiallyWeightedAverageVariance == null) {
					exponentiallyWeightedAverageVariance = varianceColumnVector.dup();
				} else {
					exponentiallyWeightedAverageVariance.asEditableMatrix().muli(beta)
							.addiColumnVector(varianceColumnVectorMul1MinusBeta);
				}
				batchNormDirectedAxonsComponent
						.setExponentiallyWeightedAverageInputFeatureVariances(exponentiallyWeightedAverageVariance);
			}
		}

		NeuronsActivation leftToRightPostDropoutInput = leftToRightAxonsActivation.getPostDropoutInput().get();

		Matrix xhat = leftToRightPostDropoutInput.getActivations(axonsContext.getMatrixFactory());
		Matrix dout = outerGradient.getOutput().getActivations(axonsContext.getMatrixFactory());

		/**
		 * . xhat:1000:101COLUMNS_SPAN_FEATURE_SET dout:100:1000ROWS_SPAN_FEATURE_SET
		 * 
		 * 
		 * 
		 */

		// System.out.println(
		// "xhat:" + xhat.getRows() + ":" + xhat.getColumns() +
		// xhatn.getFeatureOrientation());
		// Matrix dbeta = outerGradient.
		// System.out.println(
		// "dout:" + dout.getRows() + ":" + dout.getColumns()
		// + outerGradient.getFeatureOrientation());

		try (InterrimMatrix xhatMulDout = xhat.mul(dout).asInterrimMatrix()) {

			Matrix dgammaColumnVector = xhatMulDout.rowSums();

			if (axonsContext.getRegularisationLambda() != 0) {

				// LOGGER.debug("Calculating total regularisation Gradients");

				try (InterrimMatrix connectionWeightsCopy = directedAxonsComponent.getAxons().getScaleColumnVector()
						.asInterrimMatrix()) {
					Matrix regularisationAddition1 = connectionWeightsCopy.asEditableMatrix()
							.muli(axonsContext.getRegularisationLambda());

					dgammaColumnVector.asEditableMatrix().addi(regularisationAddition1);

				}
			}

			// gamma, xhat, istd = cache
			// N, _ = dout.shape

			// dbeta = np.sum(dout, axis=0)
			// dgamma = np.sum(xhat * dout, axis=0)
			// dx = (gamma*istd/N) * (N*dout - xhat*dgamma - dbeta)

			// return dx, dgamma, dbeta

			// System.out.println("dgamma:" + dgamma.getRows() + ":" + dgamma.getColumns());
			// System.out.println("dbeta:" + dbeta.getRows() + ":" + dbeta.getColumns());
			// System.out.println("xhat:" + xhat.getRows() + ":" + xhat.getColumns());
			/*
			 * Matrix dgammabTranspose =
			 * axonsContext.getMatrixFactory().createMatrix(xhatTranspose.getRows(),
			 * xhatTranspose.getColumns()); for (int i = 0; i < xhatTranspose.getColumns();
			 * i++) { dgammabTranspose.putColumn(i, dgammaTranspose); }
			 */

			// Matrix dbeta = doutTranspose.rowSums();
			// Matrix dbetaTranspose = dbeta.transpose();
			Matrix dbetaColumnVector = dout.rowSums();

			if (axonsContext.getRegularisationLambda() != 0) {

				// LOGGER.debug("Calculating total regularisation Gradients");

				try (InterrimMatrix biasesCopy = directedAxonsComponent.getAxons().getShiftColumnVector().dup()
						.asInterrimMatrix()) {

					dbetaColumnVector.asEditableMatrix()
							.addi(biasesCopy.asEditableMatrix().muli(axonsContext.getRegularisationLambda()));

				}
			}

			/*
			 * Matrix dbetabTranspose =
			 * axonsContext.getMatrixFactory().createMatrix(xhatTranspose.getRows(),
			 * xhatTranspose.getColumns()); for (int i = 0; i < xhatTranspose.getColumns();
			 * i++) { dbetabTranspose.putColumn(i, dbetaTranspose);
			 */

			int num = xhat.getColumns();

			try (InterrimMatrix istdColumnVector = axonsContext.getMatrixFactory()
					.createOnes(varianceColumnVector.getRows(), 1).asEditableMatrix()
					.divi(getStdDevColumnVector(varianceColumnVector)).asInterrimMatrix()) {

				Matrix gammaColumn = directedAxonsComponent.getAxons().getScaleColumnVector();

				try (InterrimMatrix xhatMulDGamma = xhat.mulColumnVector(dgammaColumnVector).asInterrimMatrix()) {
					try (InterrimMatrix gammaMulIstdDiviNum = gammaColumn.mulColumnVector(istdColumnVector)
							.asEditableMatrix().divi(num).asInterrimMatrix()) {
						Matrix dx = dout.mul(num).asEditableMatrix().subi(xhatMulDGamma)
								.subiColumnVector(dbetaColumnVector).muliColumnVector(gammaMulIstdDiviNum);

						NeuronsActivation dxn = new NeuronsActivationImpl(outerGradient.getOutput().getNeurons(), dx,
								outerGradient.getOutput().getFormat());

						// outerGradient.getOutput().close();

						leftToRightPostDropoutInput.close();
						xhat.close();
						// TODO
						if (!this.leftToRightAxonsActivation.getPostDropoutOutput().isImmutable()) {
							this.leftToRightAxonsActivation.getPostDropoutOutput().close();
						}
						return new DirectedComponentGradientImpl<>(outerGradient.getTotalTrainableAxonsGradients(),
								() -> new AxonsGradientImpl(directedAxonsComponent.getAxons(), dgammaColumnVector,
										dbetaColumnVector),
								dxn);
					}

				}

			}

		}
	}

	@Override
	public float getTotalRegularisationCost() {
		float totalRegularisationCost = 0f;
		if (axonsContext.getRegularisationLambda() != 0) {

			// LOGGER.info("Calculating total regularisation cost");

			if (directedAxonsComponent.getAxons() instanceof TrainableAxons) {
				try (InterrimMatrix weightsWithoutBiases = directedAxonsComponent.getAxons().getDetachedAxonWeights()
						.getConnectionWeights().getWeights().asInterrimMatrix()) {
					try (InterrimMatrix biases = directedAxonsComponent.getAxons().getDetachedAxonWeights()
							.getLeftToRightBiases().getWeights().asInterrimMatrix()) {
						float regularisationCostForWeights = weightsWithoutBiases.asEditableMatrix()
								.muli(weightsWithoutBiases).sum();
						float regularisationCostForBiases = biases.asEditableMatrix().muli(biases).sum();
						totalRegularisationCost = totalRegularisationCost + ((axonsContext.getRegularisationLambda())
								* (regularisationCostForBiases + regularisationCostForWeights)) / 2f;
					}
				}
			}
		}
		return totalRegularisationCost;
	}

	@Override
	protected DirectedComponentGradientImpl<NeuronsActivation> createBackPropagatedGradient(
			AxonsActivation rightToLeftGradientActivation, List<Supplier<AxonsGradient>> previousAxonsGradients,
			Supplier<AxonsGradient> thisAxonsGradient) {

		Matrix exponentiallyWeightedAverageMean = batchNormDirectedAxonsComponent
				.getExponentiallyWeightedAverageInputFeatureMeans();

		float beta = batchNormDirectedAxonsComponent.getBetaForExponentiallyWeightedAverages();

		try (InterrimMatrix varianceColumnVectorMul1MinusBeta = varianceColumnVector.mul(1 - (float) beta)
				.asInterrimMatrix()) {

			try (InterrimMatrix meanColumnVectorMul1MinusBeta = meanColumnVector.mul(1 - (float) beta)
					.asInterrimMatrix()) {

				if (exponentiallyWeightedAverageMean == null) {
					exponentiallyWeightedAverageMean = meanColumnVector.dup();
				} else {
					exponentiallyWeightedAverageMean.asEditableMatrix().muli(beta).addi(meanColumnVectorMul1MinusBeta);
				}
				batchNormDirectedAxonsComponent
						.setExponentiallyWeightedAverageInputFeatureMeans(exponentiallyWeightedAverageMean);

				Matrix exponentiallyWeightedAverageVariance = batchNormDirectedAxonsComponent
						.getExponentiallyWeightedAverageInputFeatureVariances();

				if (exponentiallyWeightedAverageVariance == null) {
					exponentiallyWeightedAverageVariance = varianceColumnVector.dup();
				} else {
					exponentiallyWeightedAverageVariance.asEditableMatrix().muli(beta)
							.addiColumnVector(varianceColumnVectorMul1MinusBeta);
				}
				batchNormDirectedAxonsComponent
						.setExponentiallyWeightedAverageInputFeatureVariances(exponentiallyWeightedAverageVariance);
			}
		}

		Matrix xhat = leftToRightAxonsActivation.getPostDropoutInput().get()
				.getActivations(axonsContext.getMatrixFactory());
		Matrix dout = rightToLeftGradientActivation.getPostDropoutOutput()
				.getActivations(axonsContext.getMatrixFactory());

		/**
		 * . xhat:1000:101COLUMNS_SPAN_FEATURE_SET dout:100:1000ROWS_SPAN_FEATURE_SET
		 * 
		 * 
		 * 
		 */

		// System.out.println(
		// "xhat:" + xhat.getRows() + ":" + xhat.getColumns() +
		// xhatn.getFeatureOrientation());
		// Matrix dbeta = outerGradient.
		// System.out.println(
		// "dout:" + dout.getRows() + ":" + dout.getColumns()
		// + outerGradient.getFeatureOrientation());

		try (InterrimMatrix xhatMulDout = xhat.mul(dout).asInterrimMatrix()) {

			Matrix dgammaColumnVector = xhatMulDout.rowSums();

			if (axonsContext.getRegularisationLambda() != 0) {

				// LOGGER.debug("Calculating total regularisation Gradients");

				try (InterrimMatrix connectionWeightsCopy = directedAxonsComponent.getAxons().getScaleColumnVector()
						.asInterrimMatrix()) {
					Matrix regularisationAddition1 = connectionWeightsCopy.asEditableMatrix()
							.muli(axonsContext.getRegularisationLambda());

					dgammaColumnVector.asEditableMatrix().addi(regularisationAddition1);

				}
			}

			// gamma, xhat, istd = cache
			// N, _ = dout.shape

			// dbeta = np.sum(dout, axis=0)
			// dgamma = np.sum(xhat * dout, axis=0)
			// dx = (gamma*istd/N) * (N*dout - xhat*dgamma - dbeta)

			// return dx, dgamma, dbeta

			// System.out.println("dgamma:" + dgamma.getRows() + ":" + dgamma.getColumns());
			// System.out.println("dbeta:" + dbeta.getRows() + ":" + dbeta.getColumns());
			// System.out.println("xhat:" + xhat.getRows() + ":" + xhat.getColumns());
			/*
			 * Matrix dgammabTranspose =
			 * axonsContext.getMatrixFactory().createMatrix(xhatTranspose.getRows(),
			 * xhatTranspose.getColumns()); for (int i = 0; i < xhatTranspose.getColumns();
			 * i++) { dgammabTranspose.putColumn(i, dgammaTranspose); }
			 */

			// Matrix dbeta = doutTranspose.rowSums();
			// Matrix dbetaTranspose = dbeta.transpose();
			Matrix dbetaColumnVector = dout.rowSums();

			if (axonsContext.getRegularisationLambda() != 0) {

				// LOGGER.debug("Calculating total regularisation Gradients");

				try (InterrimMatrix biasesCopy = directedAxonsComponent.getAxons().getShiftColumnVector().dup()
						.asInterrimMatrix()) {

					dbetaColumnVector.asEditableMatrix()
							.addi(biasesCopy.asEditableMatrix().muli(axonsContext.getRegularisationLambda()));

				}
			}

			/*
			 * Matrix dbetabTranspose =
			 * axonsContext.getMatrixFactory().createMatrix(xhatTranspose.getRows(),
			 * xhatTranspose.getColumns()); for (int i = 0; i < xhatTranspose.getColumns();
			 * i++) { dbetabTranspose.putColumn(i, dbetaTranspose);
			 */

			int num = xhat.getColumns();

			try (InterrimMatrix istdColumnVector = axonsContext.getMatrixFactory()
					.createOnes(varianceColumnVector.getRows(), 1).asEditableMatrix()
					.divi(getStdDevColumnVector(varianceColumnVector)).asInterrimMatrix()) {

				Matrix gammaColumn = directedAxonsComponent.getAxons().getScaleColumnVector();

				try (InterrimMatrix xhatMulDGamma = xhat.mulColumnVector(dgammaColumnVector).asInterrimMatrix()) {
					try (InterrimMatrix gammaMulIstdDiviNum = gammaColumn.mulColumnVector(istdColumnVector)
							.asEditableMatrix().divi(num).asInterrimMatrix()) {
						Matrix dx = dout.mul(num).asEditableMatrix().subi(xhatMulDGamma)
								.subiColumnVector(dbetaColumnVector).muliColumnVector(gammaMulIstdDiviNum);

						NeuronsActivation dxn = new NeuronsActivationImpl(
								rightToLeftGradientActivation.getPostDropoutOutput().getNeurons(), dx,
								rightToLeftGradientActivation.getPostDropoutOutput().getFormat());

						// outerGradient.getOutput().close();

						return new DirectedComponentGradientImpl<>(previousAxonsGradients,
								() -> new AxonsGradientImpl(directedAxonsComponent.getAxons(), dgammaColumnVector,
										dbetaColumnVector),
								dxn);
					}

				}

			}
		}

		// TODO
	}

	@Override
	protected Optional<AxonsGradient> getCalculatedAxonsGradient(AxonsActivation rightToLeftAxonsActivation) {
		throw new UnsupportedOperationException();
		// return Optional.of(new AxonsGradientImpl(directedAxonsComponent.getAxons(),
		// dgammaColumnVector, dbetaColumnVector));
	}

	@Override
	public void close(DirectedComponentActivationLifecycle completedLifeCycleStage) {
		if (completedLifeCycleStage == DirectedComponentActivationLifecycle.FORWARD_PROPAGATION) {
			close(output);
			close(leftToRightAxonsActivation.getPostDropoutOutput());
		}
	}

	private void close(NeuronsActivation activation) {
		if (!activation.isImmutable()) {
			activation.close();
		}
	}
}
