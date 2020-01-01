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
import java.util.List;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatch;
import org.ml4j.nn.components.manytomany.DefaultDirectedComponentChainBatchActivation;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponent;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.manytoone.PathCombinationStrategy;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponent;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponent;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DefaultDirectedComponentChainBipoleGraphImpl extends DefaultDirectedComponentChainBipoleGraphBase implements DefaultDirectedComponentBipoleGraph {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedComponentChainBipoleGraphImpl.class);

	private DirectedComponentFactory directedComponentFactory;
	private OneToManyDirectedComponent<?> oneToManyDirectedComponent;
	private ManyToOneDirectedComponent<?> manyToOneDirectedComponent;
	private PathCombinationStrategy pathCombinationStrategy;
	
	public DefaultDirectedComponentChainBipoleGraphImpl(DirectedComponentFactory directedComponentFactory, Neurons inputNeurons, Neurons outputNeurons, 
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch, PathCombinationStrategy pathCombinationStrategy) {
		super(inputNeurons, outputNeurons, parallelComponentChainsBatch);
		this.oneToManyDirectedComponent = directedComponentFactory.createOneToManyDirectedComponent(() -> parallelComponentChainsBatch.getComponents().size());
		this.manyToOneDirectedComponent = directedComponentFactory.createManyToOneDirectedComponent(pathCombinationStrategy);
		this.pathCombinationStrategy = pathCombinationStrategy;
	}

	@Override
	public DefaultDirectedComponentChainBatch getEdges() {
		throw new UnsupportedOperationException();
	}
	
	@Override
	public DefaultDirectedComponentBipoleGraphActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		
		if (neuronsActivation.getFeatureCount() != getInputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalStateException(neuronsActivation.getFeatureCount() + ":" + getInputNeurons().getNeuronCountExcludingBias());
		}
		LOGGER.debug("Forward propagating through DefaultDirectedComponentChainBipoleGraphImpl");
		
		if (parallelComponentChainsBatch.getComponents().size() == 1) {
			DefaultDirectedComponentChainBatchActivation parallelChainsActivation = parallelComponentChainsBatch.forwardPropagate(Arrays.asList(neuronsActivation), context);
			return new DefaultDirectedComponentBipoleGraphActivationImpl(this, parallelChainsActivation, parallelChainsActivation.getOutput().get(0));

		} else {
	
			OneToManyDirectedComponentActivation oneToManyDirectedComponentActivation = oneToManyDirectedComponent.forwardPropagate(neuronsActivation, context);
			
			DefaultDirectedComponentChainBatchActivation parallelChainsActivation = parallelComponentChainsBatch.forwardPropagate(oneToManyDirectedComponentActivation.getOutput(), context);
			
			ManyToOneDirectedComponentActivation manyToOneDirectedComponentActivation = manyToOneDirectedComponent.forwardPropagate(parallelChainsActivation.getOutput(), context);
			if (manyToOneDirectedComponentActivation.getOutput().getFeatureCount() != getOutputNeurons().getNeuronCountExcludingBias()) {
				throw new IllegalArgumentException("Many to one activation feature count of:" + manyToOneDirectedComponentActivation.getOutput().getFeatureCount() + " does not match output neuron count of:" + getOutputNeurons().getNeuronCountExcludingBias() );
			}
				
			return new DefaultDirectedComponentBipoleGraphActivationImpl(this, oneToManyDirectedComponentActivation, parallelChainsActivation, manyToOneDirectedComponentActivation);
			
		}
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public DefaultDirectedComponentBipoleGraph dup() {
		return new DefaultDirectedComponentChainBipoleGraphImpl(directedComponentFactory, inputNeurons, outputNeurons, parallelComponentChainsBatch.dup(),
				pathCombinationStrategy);
	}

}
