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
package org.ml4j.nn.activationfunctions;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;

/**
 * Default implementation of a Relu (pseudo-differentiable) activation function
 * 
 * @author Michael Lavelle
 */
public class DefaultReluActivationFunctionImpl implements DifferentiableActivationFunction {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	@Override
	public DifferentiableActivationFunctionActivation activate(NeuronsActivation activation,
			NeuronsActivationContext context) {
		NeuronsActivation input = context.isTrainingContext() ? activation.dup() : activation;
		NeuronsActivation output = activation;
		output.applyValueModifier(v -> v < 0, v -> 0);
		return new DefaultDifferentiableActivationFunctionActivationImpl(this, input, output);
	}

	@Override
	public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation activation,
			NeuronsActivationContext context) {
		NeuronsActivation output = activation.getInput();
		output.applyValueModifier(v -> true, v -> v <= 0 ? 0 : 1);
		return output;
	}

	@Override
	public ActivationFunctionType getActivationFunctionType() {
		return ActivationFunctionType.getBaseType(ActivationFunctionBaseType.LINEAR);
	}

	@Override
	public Optional<NeuronsActivationFeatureOrientation> optimisedFor() {
		return Optional.empty();
	}

	@Override
	public List<NeuronsActivationFeatureOrientation> supports() {
		return Arrays.asList(NeuronsActivationFeatureOrientation.values());
	}
}
