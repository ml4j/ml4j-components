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
package org.ml4j.nn.components.manytoone.legacy;

import java.util.ArrayList;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultManyToOneDirectedComponentActivation extends ManyToOneDirectedComponentActivationBase implements ManyToOneDirectedComponentActivation {

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultManyToOneDirectedComponentActivation.class);
	
	private int size;
	
	public DefaultManyToOneDirectedComponentActivation(int size, NeuronsActivation output) {
		super(output);
		this.size = size;
	}
	
	@Override
	public DirectedComponentGradient<List<NeuronsActivation>> backPropagate(
			DirectedComponentGradient<NeuronsActivation> gradient) {
		LOGGER.debug("Mock back propagating single neurons activations into multiple back propagated neurons activations");
		List<NeuronsActivation> activations = new ArrayList<>();
		for (int i = 0; i < size; i++) {
			activations.add(gradient.getOutput());
		}
		return new DirectedComponentGradientImpl<>(activations);
	}

}
