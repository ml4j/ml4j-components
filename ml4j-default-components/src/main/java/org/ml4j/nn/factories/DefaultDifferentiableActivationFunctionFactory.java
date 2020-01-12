/*
 * Copyright 2019 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.factories;

import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.DefaultLinearActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DefaultReluActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DefaultSigmoidActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DefaultSoftmaxActivationFunctionImpl;
import org.ml4j.nn.activationfunctions.DifferentiableActivationFunction;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;

/**
 * Default factory for different types of DifferentiableActivationFunction.
 * 
 * @author Michael Lavelle
 *
 */
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

	@Override
	public DifferentiableActivationFunction createActivationFunction(ActivationFunctionType activationFunctionType) {
		if (ActivationFunctionBaseType.LINEAR.equals(activationFunctionType.getBaseType())) {
			return createLinearActivationFunction();
		} else if (ActivationFunctionBaseType.RELU.equals(activationFunctionType.getBaseType())) {
			return createReluActivationFunction();
		} else if (ActivationFunctionBaseType.SIGMOID.equals(activationFunctionType.getBaseType())) {
			return createSigmoidActivationFunction();
		} else if (ActivationFunctionBaseType.SOFTMAX.equals(activationFunctionType.getBaseType())) {
			return createSoftmaxActivationFunction();
		} else {
			throw new IllegalArgumentException("Unsupported activation function type:" + activationFunctionType);
		}
	}

}
