package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.EditableMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class WeightedAxonsBase<L extends Neurons, R extends Neurons, A extends TrainableAxons<L, R, A>>
		implements TrainableAxons<L, R, A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(WeightedAxonsBase.class);

	protected L leftNeurons;
	protected R rightNeurons;
	protected AxonWeights axonWeights;

	public WeightedAxonsBase(L leftNeurons, R rightNeurons, AxonWeights axonWeights) {
		super();
		this.axonWeights = axonWeights;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}

	@Override
	public L getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public R getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		AxonsDropoutMask axonsDropoutMask = getLeftToRightAxonsDropoutMask(leftNeuronsActivation,
				previousRightToLeftActivation, axonsContext);

		Matrix axonWeightsInput = leftNeuronsActivation.getActivations(axonsContext.getMatrixFactory());

		if (axonsDropoutMask != null && axonsDropoutMask.getType() == AxonsDropoutMaskType.INPUT) {
			LOGGER.debug("Applying left to right input dropout mask and scaling");
			axonWeightsInput.asEditableMatrix().muli(axonsDropoutMask.getDropoutMask());
			axonWeightsInput.asEditableMatrix().muli(getLeftInputPostDropoutScaling(axonsContext));
		}

		Matrix output = axonWeights.applyToLeftToRightInput(axonWeightsInput).asEditableMatrix();

		if (axonsDropoutMask != null && axonsDropoutMask.getType() == AxonsDropoutMaskType.OUTPUT) {
			throw new UnsupportedOperationException("Left to right output dropout not yet supported");
		}

		NeuronsActivation outputActivation = new NeuronsActivationImpl(rightNeurons, output,
				leftNeuronsActivation.getFormat());

		if (!axonsContext.isTrainingContext() && !leftNeuronsActivation.isImmutable()) {
			leftNeuronsActivation.close();
		} else {
			leftNeuronsActivation.setImmutable(true);
		}

		return new AxonsActivationImpl(this, axonsDropoutMask, () -> leftNeuronsActivation, outputActivation,
				leftNeurons, rightNeurons);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		Matrix output = axonWeights
				.applyToRightToLeftInput(rightNeuronsActivation.getActivations(axonsContext.getMatrixFactory()))
				.asEditableMatrix();

		if (previousLeftToRightActivation != null && previousLeftToRightActivation.getDropoutMask() != null
				&& previousLeftToRightActivation.getDropoutMask().getType() == AxonsDropoutMaskType.INPUT) {
			LOGGER.debug("Applying right to left output dropout mask");
			output.asEditableMatrix().muli(previousLeftToRightActivation.getDropoutMask().getDropoutMask());
		}

		rightNeuronsActivation.setImmutable(true);

		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, new NeuronsActivationImpl(leftNeurons,
				output, rightNeuronsActivation.getFormat()), leftNeurons, rightNeurons);
	}

	@Override
	public boolean isTrainable(AxonsContext axonsContext) {
		return !axonsContext.isWithFreezeOut();
	}

	@Override
	public void adjustAxonWeights(AxonWeightsAdjustment adjustment,
			AxonWeightsAdjustmentDirection adjustmentDirection) {
		axonWeights.adjustWeights(adjustment, adjustmentDirection);
	}

	@Override
	public AxonWeights getDetachedAxonWeights() {
		return axonWeights.dup();
	}

	/**
	 * Return the scaling required due to left-hand side input dropout.
	 * 
	 * @param axonsContext The axons context.
	 * @return The post dropout input scaling factor.
	 */
	protected float getLeftInputPostDropoutScaling(AxonsContext axonsContext) {
		float dropoutKeepProbability = axonsContext.getLeftHandInputDropoutKeepProbability();
		if (dropoutKeepProbability == 0) {
			throw new IllegalArgumentException("Dropout keep probability cannot be set to 0");
		}
		return 1f / dropoutKeepProbability;
	}

	private AxonsDropoutMask getLeftToRightAxonsDropoutMask(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		Matrix outputDropoutMask = null;
		Matrix inputDropoutMask = createLeftInputDropoutMask(leftNeuronsActivation, axonsContext);

		AxonsDropoutMask previousInputDropoutMask = previousRightToLeftActivation == null
				|| previousRightToLeftActivation.getDropoutMask() == null ? null
						: previousRightToLeftActivation.getDropoutMask();
		if (previousInputDropoutMask != null && previousInputDropoutMask.getType() == AxonsDropoutMaskType.INPUT
				&& previousInputDropoutMask.getDropoutMask() != null) {
			LOGGER.info("Using previous right to left input dropout mask as left to right output dropout mask");
			outputDropoutMask = previousInputDropoutMask.getDropoutMask();
		}

		AxonsDropoutMask axonsDropoutMask = null;
		if (outputDropoutMask != null && inputDropoutMask != null) {
			throw new IllegalStateException();
		} else if (outputDropoutMask != null) {
			axonsDropoutMask = new AxonsDropoutMaskImpl(outputDropoutMask, AxonsDropoutMaskType.OUTPUT);
		} else if (inputDropoutMask != null) {
			axonsDropoutMask = new AxonsDropoutMaskImpl(inputDropoutMask, AxonsDropoutMaskType.INPUT);
		}

		return axonsDropoutMask;

	}

	/**
	 * Return the dropout mask for left hand side input.
	 * 
	 * @param axonsContext The axons context
	 * @return The input dropout mask applied at the left hand side of these Axons
	 */
	protected Matrix createLeftInputDropoutMask(NeuronsActivation leftNeuronsActivation, AxonsContext axonsContext) {

		double leftHandInputDropoutKeepProbability = axonsContext.getLeftHandInputDropoutKeepProbability();
		if (leftHandInputDropoutKeepProbability == 1 || !axonsContext.isTrainingContext()) {
			return null;
		} else {
			if (!isLeftInputDropoutSupported()) {
				throw new IllegalStateException("Left input dropout is not supported for these axons");
			}

			LOGGER.debug("Creating left input dropout mask");

			EditableMatrix dropoutMask = axonsContext.getMatrixFactory()
					.createZeros(leftNeuronsActivation.getRows(), leftNeuronsActivation.getColumns())
					.asEditableMatrix();
			for (int i = 0; i < dropoutMask.getRows(); i++) {
				for (int j = 0; j < dropoutMask.getColumns(); j++) {
					if (Math.random() < leftHandInputDropoutKeepProbability) {
						dropoutMask.put(i, j, 1);
					}
				}
			}
			return dropoutMask;

		}
	}

	protected abstract boolean isLeftInputDropoutSupported();

	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}
}
