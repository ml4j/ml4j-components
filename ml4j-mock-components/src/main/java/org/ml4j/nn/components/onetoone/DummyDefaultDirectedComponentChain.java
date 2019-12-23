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
package org.ml4j.nn.components.onetoone;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mock implementation of DefaultDirectedComponentChain.
 * 
 * Encapsulates a mock sequential chain of DefaultChainableDirectedComponents
 * 
 * @author Michael Lavelle
 */
public class DummyDefaultDirectedComponentChain extends DefaultDirectedComponentChainBase implements DefaultDirectedComponentChain {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDefaultDirectedComponentChain.class);

	/**
	 * DummyDefaultDirectedComponentChain constructor
	 * 
	 * @param sequentialComponents The list of DefaultChainableDirectedComponents with which to initialise this chain.
	 */
	public DummyDefaultDirectedComponentChain(List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		super(sequentialComponents);
	}

	@Override
	public DefaultDirectedComponentChainActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		LOGGER.debug("Mock forward propagating through DummyDefaultDirectedComponentChain");
		NeuronsActivation inFlightActivation = neuronsActivation;
		List<DefaultChainableDirectedComponentActivation> activations = new ArrayList<>();
		int index = 0;
		for (DefaultChainableDirectedComponent<?, ?> component : sequentialComponents) {
			DefaultChainableDirectedComponentActivation activation = forwardPropagate(inFlightActivation, component, index, context);
			activations.add(activation);
			inFlightActivation = activation.getOutput();
			index++;
		}
		
		return new DummyDefaultDirectedComponentChainActivation(this, inFlightActivation);
	}

	@Override
	public DefaultDirectedComponentChain dup() {
		return new DummyDefaultDirectedComponentChain(sequentialComponents.stream().map(c -> c.dup()).collect(Collectors.toList()));
	}
}
