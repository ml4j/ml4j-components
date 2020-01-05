package org.ml4j.nn.activationfunctions;

import org.ml4j.nn.activationfunctions.base.DifferentiableActivationFunctionActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDifferentiableActivationFunctionActivationImpl extends DifferentiableActivationFunctionActivationBase implements DifferentiableActivationFunctionActivation {

	public DefaultDifferentiableActivationFunctionActivationImpl(DifferentiableActivationFunction activationFunction,
			NeuronsActivation input, NeuronsActivation output) {
		super(activationFunction, input, output);
	}
}
