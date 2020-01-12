package org.ml4j.nn.activationfunctions;

import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;

public class NoOpDifferentiableActivationFunction implements DifferentiableActivationFunction {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private ActivationFunctionType activationFunctionType;
	
	public NoOpDifferentiableActivationFunction(ActivationFunctionType activationFunctionType) {
		this.activationFunctionType = activationFunctionType;
	}

	@Override
	public DifferentiableActivationFunctionActivation activate(NeuronsActivation input, NeuronsActivationContext arg1) {
		return new NoOpDifferentiableActivationFunctionActivation(this, input);
	}

	@Override
	public ActivationFunctionType getActivationFunctionType() {
		return activationFunctionType;
	}

	@Override
	public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation arg0,
			NeuronsActivationContext arg1) {
		return arg0.getInput();
	}

}
