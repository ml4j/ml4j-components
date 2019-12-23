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
package org.ml4j.nn.components.activationfunctions;

import org.ml4j.nn.activationfunctions.DifferentiableActivationFunctionComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.activationfunctions.base.DifferentiableActivationFunctionComponentActivationBase;
import org.ml4j.nn.costfunctions.CostFunctionGradient;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Dummy implementation of DifferentiableActivationFunctionActivation, encapsulating the activations from a DummyDifferentiableActivationFunction
 * and providing the logic required in order to back propagate gradients back through the activations.
 * 
 * @author Michael Lavelle
 */
public class DefaultDifferentiableActivationFunctionComponentActivationImpl extends DifferentiableActivationFunctionComponentActivationBase implements DifferentiableActivationFunctionComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDifferentiableActivationFunctionComponentActivationImpl.class);
		
	/**
	 * @param input The input to the DummyDifferentiableActivationFunction
	 * @param output The output from the DummyDifferentiableActivationFunction
	 */
	public DefaultDifferentiableActivationFunctionComponentActivationImpl(DifferentiableActivationFunctionComponent activationFunctionComponent, NeuronsActivation input, NeuronsActivation output) {
		super(activationFunctionComponent, input, output);
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		LOGGER.debug("Back propagating gradient through DummyDifferentiableActivationFunctionActivation");
		return gradient;
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(CostFunctionGradient costFunctionGradient) {
		LOGGER.debug("Back propagating cost function gradient through DummyDifferentiableActivationFunctionActivation");
		return new DirectedComponentGradientImpl<>(output);
	}
}
