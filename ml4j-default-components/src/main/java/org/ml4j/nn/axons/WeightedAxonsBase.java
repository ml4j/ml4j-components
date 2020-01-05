package org.ml4j.nn.axons;

import org.ml4j.Matrix;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.ImageNeuronsActivationImpl;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public abstract class WeightedAxonsBase<L extends Neurons, R extends Neurons, A extends TrainableAxons<L, R, A>> implements TrainableAxons<L, R, A> {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

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
		
		Matrix output = axonWeights.applyToLeftToRightInput(leftNeuronsActivation.getActivations(axonsContext.getMatrixFactory()))
				.asEditableMatrix();
		
		NeuronsActivation outputActivation = null;
		if (leftNeuronsActivation instanceof ImageNeuronsActivation && rightNeurons instanceof Neurons3D) {
			outputActivation = new ImageNeuronsActivationImpl(output, (Neurons3D) this.rightNeurons, leftNeuronsActivation.getFeatureOrientation(), false);
		} else {
			outputActivation = new NeuronsActivationImpl(output, leftNeuronsActivation.getFeatureOrientation());
		}
		
		return new AxonsActivationImpl(this, null, () -> leftNeuronsActivation,
				outputActivation,
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
