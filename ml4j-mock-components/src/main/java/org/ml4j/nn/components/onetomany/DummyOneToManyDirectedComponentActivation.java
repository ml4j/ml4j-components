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
package org.ml4j.nn.components.onetomany;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.onetomany.base.OneToManyDirectedComponentActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mock implementation of OneToManyDirectedComponentActivation - encapsulating the activation from a DummyOneToManyDirectedComponent.
 * 
 * @author Michael Lavelle
 */
public class DummyOneToManyDirectedComponentActivation extends OneToManyDirectedComponentActivationBase implements OneToManyDirectedComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyOneToManyDirectedComponentActivation.class);
	
	/**
	 * DummyOneToManyDirectedComponentActivation constructor
	 * 
	 * @param input The neurons activation input to the one to many component.
	 * @param outputNeuronsActivationCount The desired number of instances of output neuron activations, one for each of the components
	 * on the RHS of the OneToManyDirectedComponentActivation.
	 */
	public DummyOneToManyDirectedComponentActivation(NeuronsActivation input, int outputNeuronsActivationCount) {
		super(input, outputNeuronsActivationCount);
	}
	
	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<List<NeuronsActivation>> gradient) {
		LOGGER.debug("Mock back propagating multiple gradient neurons activations into a single combined neurons activation");
		return new DirectedComponentGradientImpl<>(gradient.getOutput().get(0));
	}
}
