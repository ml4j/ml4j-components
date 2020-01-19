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

import org.ml4j.InterrimMatrix;
import org.ml4j.Matrix;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationContext;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of a Sigmoid differentiable activation function
 * 
 * @author Michael Lavelle
 */
public class DefaultSigmoidActivationFunctionImpl implements DifferentiableActivationFunction {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultSigmoidActivationFunctionImpl.class);

	@Override
	public DifferentiableActivationFunctionActivation activate(NeuronsActivation input,
			NeuronsActivationContext context) {
		LOGGER.debug("Activating through SigmoidActivationFunction:" + input.getFeatureCount() + ":"
				+ +input.getExampleCount() + ":" + input.getFeatureOrientation());
		

		Matrix sigmoidOfInputActivationsMatrix = input.getActivations(context.getMatrixFactory()).sigmoid();
		NeuronsActivation output = new NeuronsActivationImpl(input.getNeurons(), sigmoidOfInputActivationsMatrix, input.getFeatureOrientation());
		output.setImmutable(true);
		return new DefaultDifferentiableActivationFunctionActivationImpl(this, input, output
				);
	}

	@Override
	public NeuronsActivation activationGradient(DifferentiableActivationFunctionActivation activationFunctionActivation,
			NeuronsActivationContext context) {

		LOGGER.debug("Performing sigmoid gradient of NeuronsActivation:");

		if (activationFunctionActivation.getActivationFunction() instanceof DefaultSigmoidActivationFunctionImpl) {
			Matrix sigmoidOfActivationInput = activationFunctionActivation.getOutput()
					.getActivations(context.getMatrixFactory());

			try (InterrimMatrix sigmoidOfActivationInputSquared = sigmoidOfActivationInput.mul(sigmoidOfActivationInput)
					.asInterrimMatrix()) {

				Matrix gradientAtActivationInput = sigmoidOfActivationInput.sub(sigmoidOfActivationInputSquared);
				return new NeuronsActivationImpl(activationFunctionActivation.getInput().getNeurons(), gradientAtActivationInput,
						activationFunctionActivation.getInput().getFeatureOrientation());
			}
			

		} else {

			Matrix activationInput = activationFunctionActivation.getInput().getActivations(context.getMatrixFactory());
			try (InterrimMatrix sigmoidOfActivationInput = activationInput.sigmoid().asInterrimMatrix()) {

				try (InterrimMatrix sigmoidOfActivationInputSquared = sigmoidOfActivationInput
						.mul(sigmoidOfActivationInput).asInterrimMatrix()) {

					Matrix gradientAtActivationInput = sigmoidOfActivationInput.sub(sigmoidOfActivationInputSquared);

					return new NeuronsActivationImpl(activationFunctionActivation.getInput().getNeurons(), gradientAtActivationInput,
							activationFunctionActivation.getInput().getFeatureOrientation());
				}
			}

		}

	}

	@Override
	public ActivationFunctionType getActivationFunctionType() {
		return ActivationFunctionType.getBaseType(ActivationFunctionBaseType.SIGMOID);
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
