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

import org.ml4j.nn.components.DirectedComponentActivationLifecycle;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DefaultDirectedComponentChain.
 * 
 * Encapsulates a sequential chain of DefaultChainableDirectedComponents
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedComponentChainImpl extends DefaultDirectedComponentChainBase
		implements DefaultDirectedComponentChain {

	/**
	 * Default serialization id
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedComponentChainImpl.class);

	/**
	 * DummyDefaultDirectedComponentChain constructor
	 * 
	 * @param sequentialComponents The list of DefaultChainableDirectedComponents
	 *                             with which to initialise this chain.
	 */
	public DefaultDirectedComponentChainImpl(List<DefaultChainableDirectedComponent<?, ?>> sequentialComponents) {
		super(sequentialComponents);
	}

	@Override
	public DefaultDirectedComponentChainActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		LOGGER.debug("Forward propagating through DefaultDirectedComponentChainImpl");
		NeuronsActivation inFlightActivation = neuronsActivation;

		if (inFlightActivation.getFeatureCount() != getInputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalStateException("Expected " + getInputNeurons().getNeuronCountExcludingBias() + " features but received " + inFlightActivation.getFeatureCount()) ;
		}
		int inFlightActivationFeatureCount = neuronsActivation.getFeatureCount();
		List<DefaultChainableDirectedComponentActivation> activations = new ArrayList<>();
		for (DefaultChainableDirectedComponent<?, ?> component : sequentialComponents) {
			DefaultChainableDirectedComponentActivation activation = forwardPropagate(inFlightActivation, component,
					context);
			activations.add(activation);
			inFlightActivation = activation.getOutput();
			inFlightActivationFeatureCount = inFlightActivation.getFeatureCount();
		}

		for (DefaultChainableDirectedComponentActivation act : activations) {
			// TODO
			if (act.getOutput() != inFlightActivation && context.isTrainingContext()) {
				act.close(DirectedComponentActivationLifecycle.FORWARD_PROPAGATION);
			}
		}

		if (inFlightActivationFeatureCount != getOutputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalStateException();
		}

		return new DefaultDirectedComponentChainActivationImpl(this, activations);
	}

	@Override
	public DefaultDirectedComponentChain dup(DirectedComponentFactory directedComponentFactory) {
		return new DefaultDirectedComponentChainImpl(
				sequentialComponents.stream().map(c -> c.dup(directedComponentFactory)).collect(Collectors.toList()));
	}

}
