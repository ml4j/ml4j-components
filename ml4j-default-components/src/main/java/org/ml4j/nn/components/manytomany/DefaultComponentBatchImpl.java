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

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.nn.components.ChainableDirectedComponent;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.base.DefaultComponentBatchBase;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.codepoetics.protonpack.StreamUtils;

/**
 * Default implementation of a batch of DefaultChainableDirectedComponent
 * instances with DirectedComponentsContext contexts that can be activated in
 * parallel.
 * 
 * @author Michael Lavelle
 */
public class DefaultComponentBatchImpl extends DefaultComponentBatchBase {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultComponentBatchImpl.class);

	public DefaultComponentBatchImpl(List<DefaultChainableDirectedComponent<?, ?>> parallelComponents) {
		super(parallelComponents);
	}

	@Override
	public DefaultDirectedComponentBatchActivation forwardPropagate(List<NeuronsActivation> neuronActivations,
			DirectedComponentsContext context) {

		LOGGER.debug("Forward propagating through DefaultComponentChainBatchImpl");

		List<DefaultChainableDirectedComponentActivation> chainActivations = StreamUtils
				.zipWithIndex(neuronActivations.parallelStream()).map(a -> forwardPropagate(a.getValue(),
						parallelComponents.get((int) a.getIndex()), (int) a.getIndex(), context))
				.collect(Collectors.toList());

		return new DefaultDirectedComponentBatchActivationImpl(chainActivations);
	}

	protected <X, Y> Y forwardPropagate(NeuronsActivation input,
			DefaultChainableDirectedComponent<? extends Y, X> component, int componentIndex,
			DirectedComponentsContext context) {
		return component.forwardPropagate(input, component.getContext(context));
	}

	@Override
	public DefaultDirectedComponentBatch dup(DirectedComponentFactory directedComponentFactory) {
		return new DefaultComponentBatchImpl(parallelComponents);
	}

	@Override
	public DirectedComponentsContext getContext(DirectedComponentsContext directedComponentsContext) {
		return directedComponentsContext;
	}

	@Override
	public List<ChainableDirectedComponent<List<NeuronsActivation>, ? extends ChainableDirectedComponentActivation<List<NeuronsActivation>>, ?, DirectedComponentFactory>> decompose() {
		return Arrays.asList(this);
	}

}
