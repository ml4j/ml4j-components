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
package org.ml4j.nn.components.manytomany;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.base.DefaultComponentBatchBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyDefaultComponentBatch extends DefaultComponentBatchBase implements DefaultDirectedComponentBatch {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDefaultComponentBatch.class);

	public DummyDefaultComponentBatch(List<DefaultChainableDirectedComponent<?, ?>> parallelComponents) {
		super(parallelComponents);
	}

	@Override
	public DefaultDirectedComponentBatchActivation forwardPropagate(List<NeuronsActivation> neuronActivations,
			DirectedComponentsContext context) {
		// TODO
		int index = 0;
		List<DefaultChainableDirectedComponentActivation> chainActivations = new ArrayList<>();
		for (NeuronsActivation neuronActivation : neuronActivations) {
			DefaultChainableDirectedComponentActivation chainActivation = forwardPropagate(neuronActivation,
					parallelComponents.get(index), context);
			chainActivations.add(chainActivation);
			index++;
		}
		LOGGER.debug("Mock forward propagating through DummyDefaultComponentChainBatch");

		return new DummyDirectedComponentBatchActivation(chainActivations);
	}

	protected <X extends Serializable, Y> Y forwardPropagate(NeuronsActivation input,
			DefaultChainableDirectedComponent<? extends Y, X> component,
			DirectedComponentsContext context) {
		return component.forwardPropagate(input, component.getContext(context));
	}

	@Override
	public DefaultDirectedComponentBatch dup(DirectedComponentFactory directedComponentFactory) {
		return new DummyDefaultComponentBatch(parallelComponents);
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext) {
		return directedComponentsContext;
	}

	@Override
	public List<DefaultDirectedComponentBatch> decompose() {
		return Arrays.asList(this);
	}

}
