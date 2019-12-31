package org.ml4j.nn.factories;

import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.activationfunctions.mocks.DummyDifferentiableActivationFunctionImpl;

public class DummyDifferentiableActivationFunctionFactory implements DifferentiableActivationFunctionFactory {

	@Override
	public DifferentiableActivationFunction createReluActivationFunction() {
		return new DummyDifferentiableActivationFunctionImpl(ActivationFunctionType.RELU, false);
	}

	@Override
	public DifferentiableActivationFunction createSigmoidActivationFunction() {
		return new DummyDifferentiableActivationFunctionImpl(ActivationFunctionType.SIGMOID, false);
	}

	@Override
	public DifferentiableActivationFunction createSoftmaxActivationFunction() {
		return new DummyDifferentiableActivationFunctionImpl(ActivationFunctionType.SOFTMAX, true);
	}

	@Override
	public DifferentiableActivationFunction createLinearActivationFunction() {
		return new DummyDifferentiableActivationFunctionImpl(ActivationFunctionType.LINEAR, true);
	}

}
