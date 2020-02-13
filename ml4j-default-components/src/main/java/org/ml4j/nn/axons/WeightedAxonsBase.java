package org.ml4j.nn.axons;

import java.util.Optional;

import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public abstract class WeightedAxonsBase<L extends Neurons, R extends Neurons, A extends TrainableAxons<L, R, A>, C extends AxonsConfig<L, R>>
		extends AxonsBase<L, R, A, C>
		implements TrainableAxons<L, R, A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@SuppressWarnings("unused")
	private static final Logger LOGGER = LoggerFactory.getLogger(WeightedAxonsBase.class);
	
	protected AxonWeights axonWeights;
	
	public WeightedAxonsBase(C axonsConfig, AxonWeights axonWeights) {
		super(axonsConfig);
		this.axonWeights = axonWeights;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {

		NeuronsActivation outputActivation = axonWeights.applyToLeftToRightInput(leftNeuronsActivation, 
				axonsContext);

		if (!axonsContext.isTrainingContext() && !leftNeuronsActivation.isImmutable()) {
			leftNeuronsActivation.close();
		} else {
			leftNeuronsActivation.setImmutable(true);
		}

		return new AxonsActivationImpl(this, null, () -> leftNeuronsActivation, outputActivation);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {

		NeuronsActivation output = axonWeights
				.applyToRightToLeftInput(rightNeuronsActivation, axonsContext);

		rightNeuronsActivation.setImmutable(true);

		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation, output);
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
