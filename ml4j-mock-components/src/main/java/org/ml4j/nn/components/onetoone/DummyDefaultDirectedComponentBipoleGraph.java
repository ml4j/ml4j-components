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

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentBatch;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentBipoleGraphBase;
import org.ml4j.nn.neurons.DummyNeuronsActivation;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.ml4j.nn.neurons.format.NeuronsActivationFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyDefaultDirectedComponentBipoleGraph extends DefaultDirectedComponentBipoleGraphBase
		implements DefaultDirectedComponentBipoleGraph {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyDefaultDirectedComponentBipoleGraph.class);

	public DummyDefaultDirectedComponentBipoleGraph(String name, Neurons inputNeurons, Neurons outputNeurons,
			DefaultDirectedComponentBatch parallelComponentChainsBatch, PathCombinationStrategy pathCombinationStrategy) {
		super(name, inputNeurons, outputNeurons, parallelComponentChainsBatch, pathCombinationStrategy);
	}

	@Override
	public DefaultDirectedComponentBatch getEdges() {
		throw new UnsupportedOperationException();
	}

	@Override
	public DefaultDirectedComponentBipoleGraphActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {

		if (neuronsActivation.getFeatureCount() != getInputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalStateException(
					neuronsActivation.getFeatureCount() + ":" + getInputNeurons().getNeuronCountExcludingBias());
		}
		LOGGER.debug("Mock forward propagating through DummyDefaultDirectedComponentChainBipoleGraph");

		return new DummyDefaultDirectedComponentBipoleGraphActivation(this, new DummyNeuronsActivation(
				getOutputNeurons(), neuronsActivation.getFeatureOrientation(), neuronsActivation.getExampleCount()));
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}
	

	@Override
	public Set<DefaultChainableDirectedComponent<?, ?>> flatten() {
		Set<DefaultChainableDirectedComponent<?, ?>> allComponentsIncludingThis = new HashSet<>(Arrays.asList(this));
		allComponentsIncludingThis.addAll(this.parallelComponentBatch.getComponents().stream().flatMap(c -> c.flatten().stream()).collect(Collectors.toSet()));
		return allComponentsIncludingThis;
	}

	@Override
	public DefaultDirectedComponentBipoleGraph dup(DirectedComponentFactory directedComponentFactory) {
		return new DummyDefaultDirectedComponentBipoleGraph(name, inputNeurons, outputNeurons, parallelComponentBatch.dup(directedComponentFactory), pathCombinationStrategy);
	}
	@Override
	public Optional<NeuronsActivationFormat<?>> optimisedFor() {
		return Optional.empty();
	}
	
	@Override
	public boolean isSupported(NeuronsActivationFormat<?> format) {
		return NeuronsActivationFeatureOrientation.ROWS_SPAN_FEATURE_SET.equals(format.getFeatureOrientation());
	}
}
