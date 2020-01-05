package org.ml4j.nn.activationfunctions;

import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationImpl;

public class DefaultLinearActivationFunctionImpl implements DifferentiableActivationFunction {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public DifferentiableActivationFunctionActivation activate(NeuronsActivation activation, NeuronsActivationContext context) {
		NeuronsActivation output = activation.dup();
		return new DefaultDifferentiableActivationFunctionActivationImpl(this, activation, output);
	}
	
	@Override
	public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation activation,
			NeuronsActivationContext context) {
		NeuronsActivation output = new NeuronsActivationImpl(context.getMatrixFactory().createOnes(activation.getInput().getFeatureCount(),
				activation.getInput().getExampleCount()), activation.getInput().getFeatureOrientation());
		return output;
	}

	@Override
	public ActivationFunctionType getActivationFunctionType() {
		return ActivationFunctionType.RELU;
	}

	

}
