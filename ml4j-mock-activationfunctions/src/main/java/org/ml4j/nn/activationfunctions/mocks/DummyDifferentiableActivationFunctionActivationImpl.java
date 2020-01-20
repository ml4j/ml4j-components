package org.ml4j.nn.activationfunctions.mocks;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionActivation;
import org.ml4j.nn.activationfunctions.base.DifferentiableActivationFunctionActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DummyDifferentiableActivationFunctionActivationImpl extends DifferentiableActivationFunctionActivationBase
		implements DifferentiableActivationFunctionActivation {

	public DummyDifferentiableActivationFunctionActivationImpl(DifferentiableActivationFunction activationFunction,
			NeuronsActivation input, NeuronsActivation output) {
		super(activationFunction, input, output);
	}
}
