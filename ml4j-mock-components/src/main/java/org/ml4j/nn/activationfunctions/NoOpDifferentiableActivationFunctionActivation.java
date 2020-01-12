package org.ml4j.nn.activationfunctions;

import org.ml4j.nn.neurons.NeuronsActivation;

public class NoOpDifferentiableActivationFunctionActivation implements DifferentiableActivationFunctionActivation {

	private DifferentiableActivationFunction activationFunction;
	private NeuronsActivation input;
	
	public NoOpDifferentiableActivationFunctionActivation(DifferentiableActivationFunction activationFunction, NeuronsActivation input) {
		this.activationFunction = activationFunction;
		this.input = input;
	}
	
	@Override
	public DifferentiableActivationFunction getActivationFunction() {
		return activationFunction;
	}

	@Override
	public NeuronsActivation getInput() {
		return input;
	}

	@Override
	public NeuronsActivation getOutput() {
		return input;
	}

}
