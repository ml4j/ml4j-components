package org.ml4j.nn.activationfunctions;

import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;

public class DefaultReluActivationFunctionImpl implements DifferentiableActivationFunction {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public DifferentiableActivationFunctionActivation activate(NeuronsActivation activation, NeuronsActivationContext context) {
		NeuronsActivation output = activation.dup();
	    output.applyValueModifier(v -> v < 0, v -> 0);
		return new DefaultDifferentiableActivationFunctionActivationImpl(this, activation, output);
	}
	
	@Override
	public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation activation,
			NeuronsActivationContext context) {
		NeuronsActivation output = activation.getInput().dup();
	    output.applyValueModifier(v -> true, v -> v <= 0 ? 0 : 1);
		return output;
	}

	@Override
	public ActivationFunctionType getActivationFunctionType() {
		return ActivationFunctionType.RELU;
	}

	

}
