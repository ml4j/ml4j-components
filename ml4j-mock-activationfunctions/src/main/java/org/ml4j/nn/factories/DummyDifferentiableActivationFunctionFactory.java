package org.ml4j.nn.factories;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.activationfunctions.mocks.DummyDifferentiableActivationFunctionImpl;

public class DummyDifferentiableActivationFunctionFactory implements DifferentiableActivationFunctionFactory {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public DifferentiableActivationFunction createReluActivationFunction() {
		return new DummyDifferentiableActivationFunctionImpl(
				ActivationFunctionType.getBaseType(ActivationFunctionBaseType.RELU), false);
	}
	
	@Override
	public DifferentiableActivationFunction createLeakyReluActivationFunction(float alpha) {
		return new DummyDifferentiableActivationFunctionImpl(
				ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LEAKYRELU), false);
	}

	@Override
	public DifferentiableActivationFunction createSigmoidActivationFunction() {
		return new DummyDifferentiableActivationFunctionImpl(
				ActivationFunctionType.getBaseType(ActivationFunctionBaseType.SIGMOID), false);
	}

	@Override
	public DifferentiableActivationFunction createSoftmaxActivationFunction() {
		return new DummyDifferentiableActivationFunctionImpl(
				ActivationFunctionType.getBaseType(ActivationFunctionBaseType.SOFTMAX), true);
	}

	@Override
	public DifferentiableActivationFunction createLinearActivationFunction() {
		return new DummyDifferentiableActivationFunctionImpl(
				ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LINEAR), true);
	}

	@Override
	public DifferentiableActivationFunction createActivationFunction(ActivationFunctionType activationFunctionType, ActivationFunctionProperties activationFunctionProperties) {
		if (ActivationFunctionBaseType.LINEAR.equals(activationFunctionType.getBaseType())) {
			return createLinearActivationFunction();
		} else if (ActivationFunctionBaseType.RELU.equals(activationFunctionType.getBaseType())) {
			return createReluActivationFunction();
		} else if (ActivationFunctionBaseType.SIGMOID.equals(activationFunctionType.getBaseType())) {
			return createSigmoidActivationFunction();
		} else if (ActivationFunctionBaseType.SOFTMAX.equals(activationFunctionType.getBaseType())) {
			return createSoftmaxActivationFunction();
		} else if (ActivationFunctionBaseType.LEAKYRELU.equals(activationFunctionType.getBaseType())) {
			if (activationFunctionProperties.getAlpha().isPresent())  {
				return createLeakyReluActivationFunction(activationFunctionProperties.getAlpha().get());
			} else {
				throw new IllegalArgumentException("ActivationFunctionProperties.alpha must be specified for LEAKYRELU activation function");
			}
		} else {
			throw new IllegalArgumentException("Unsupported activation function type:" + activationFunctionType);
		}
	}

}
