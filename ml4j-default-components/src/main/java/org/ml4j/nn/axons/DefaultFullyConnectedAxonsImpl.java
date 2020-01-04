package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DefaultFullyConnectedAxonsImpl implements FullyConnectedAxons {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private Neurons leftNeurons;
	private Neurons rightNeurons;
	private AxonWeights axonWeights;

	public DefaultFullyConnectedAxonsImpl(Neurons leftNeurons, Neurons rightNeurons, AxonWeights axonWeights) {
		super();
		this.axonWeights = axonWeights;
		this.leftNeurons = leftNeurons;
		this.rightNeurons = rightNeurons;
	}
	
	@Override
	public Neurons getLeftNeurons() {
		return leftNeurons;
	}

	@Override
	public Neurons getRightNeurons() {
		return rightNeurons;
	}

	@Override
	public AxonsActivation pushLeftToRight(NeuronsActivation leftNeuronsActivation,
			AxonsActivation previousRightToLeftActivation, AxonsContext axonsContext) {
		
		Matrix output = axonWeights.applyToLeftToRightInput(leftNeuronsActivation.getActivations(axonsContext.getMatrixFactory()))
				.asEditableMatrix();
		
		return new AxonsActivationImpl(this, null, () -> leftNeuronsActivation,
				new NeuronsActivationImpl(
						output,
						NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET),
				leftNeurons, rightNeurons);
	}

	@Override
	public AxonsActivation pushRightToLeft(NeuronsActivation rightNeuronsActivation,
			AxonsActivation previousLeftToRightActivation, AxonsContext axonsContext) {
		Matrix output = axonWeights.applyToRightToLeftInput(rightNeuronsActivation.getActivations(axonsContext.getMatrixFactory()))
				.asEditableMatrix();
		
		return new AxonsActivationImpl(this, null, () -> rightNeuronsActivation,
				new NeuronsActivationImpl(
						output,
						NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET),
				leftNeurons, rightNeurons);
	}

	@Override
	public FullyConnectedAxons dup() {
		return new DefaultFullyConnectedAxonsImpl(leftNeurons, rightNeurons, axonWeights.dup());
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

}
