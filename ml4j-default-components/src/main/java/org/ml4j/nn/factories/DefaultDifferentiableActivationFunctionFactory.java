package org.ml4j.nn.factories;

import org.ml4j.nn.activationfunctions.DefaultLinearActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DefaultReluActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DefaultSigmoidActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DefaultSoftmaxActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;

public class DefaultDifferentiableActivationFunctionFactory implements DifferentiableActivationFunctionFactory {

	@Override
	public DifferentiableActivationFunction createReluActivationFunction() {
		return new DefaultReluActivationFunctionImpl();
	}

	@Override
	public DifferentiableActivationFunction createSigmoidActivationFunction() {
		return new DefaultSigmoidActivationFunctionImpl();
	}

	@Override
	public DifferentiableActivationFunction createSoftmaxActivationFunction() {
		return new DefaultSoftmaxActivationFunctionImpl();
	}

	@Override
	public DifferentiableActivationFunction createLinearActivationFunction() {
		return new DefaultLinearActivationFunctionImpl();
	}

}
