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
import java.util.Optional;

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
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainBipoleGraphBase;
import org.ml4j.nn.neurons.Neurons;
import org.ml4j.nn.neurons.Neurons1D;
import org.ml4j.nn.neurons.Neurons3D;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationFeatureOrientation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Default implementation of DefaultDirectedComponentBipoleGraph,  consisting of a parallel edges ( DefaultDirectedComponentChainBatch ), with
 * the paths through those parallel edges all starting at the same point in the network and ending at the same point in the network.
 * 
 * The activations at the input of this graph are mapped to the DefaultDirectedComponentChainBatch activations via a OneToManyDirectedComponent,
 * and the DefaultDirectedComponentChainBatch output activations are mapped to the output of this graph via a ManyToOneDirectedComponent, using
 * a specified PathCombinationStrategy.
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedComponentChainBipoleGraphImpl extends DefaultDirectedComponentChainBipoleGraphBase implements DefaultDirectedComponentChainBipoleGraph {

	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private static final Logger LOGGER = LoggerFactory.getLogger(DefaultDirectedComponentChainBipoleGraphImpl.class);

	private OneToManyDirectedComponent<?> oneToManyDirectedComponent;
	private ManyToOneDirectedComponent<?> manyToOneDirectedComponent;
	private PathCombinationStrategy pathCombinationStrategy;
	
	
	/**
	 * 
	 * @param directedComponentFactory A DirectedComponentFactory instance, used to construct the nested OneToManyDirectedComponent and ManyToOneDirectedComponent.
	 * @param inputNeurons The input neurons of this graph.
	 * @param outputNeurons The output neurons of this graph.
	 * @param parallelComponentChainsBatch The batch of parallel edges within this graph, connecting
	 * @param pathCombinationStrategy The strategy specifying how the outputs of the parallel edges should be combined to
	 * produce the output activations.
	 */
	public DefaultDirectedComponentChainBipoleGraphImpl(DirectedComponentFactory directedComponentFactory, Neurons inputNeurons, Neurons1D outputNeurons, 
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch, PathCombinationStrategy pathCombinationStrategy) {
		this(inputNeurons, outputNeurons, parallelComponentChainsBatch, directedComponentFactory.createOneToManyDirectedComponent(() -> parallelComponentChainsBatch.getComponents().size()), 
				directedComponentFactory.createManyToOneDirectedComponent(outputNeurons, pathCombinationStrategy), pathCombinationStrategy);
		this.oneToManyDirectedComponent = directedComponentFactory.createOneToManyDirectedComponent(() -> parallelComponentChainsBatch.getComponents().size());
		this.manyToOneDirectedComponent = directedComponentFactory.createManyToOneDirectedComponent(outputNeurons, pathCombinationStrategy);
		this.pathCombinationStrategy = pathCombinationStrategy;
	}

	/**
	 * 
	 * @param directedComponentFactory A DirectedComponentFactory instance, used to construct the nested OneToManyDirectedComponent and ManyToOneDirectedComponent.
	 * @param inputNeurons The input neurons of this graph.
	 * @param outputNeurons The output neurons of this graph.
	 * @param parallelComponentChainsBatch The batch of parallel edges within this graph, connecting
	 * @param pathCombinationStrategy The strategy specifying how the outputs of the parallel edges should be combined to
	 * produce the output activations.
	 */
	public DefaultDirectedComponentChainBipoleGraphImpl(DirectedComponentFactory directedComponentFactory, Neurons inputNeurons, Neurons3D outputNeurons, 
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch, PathCombinationStrategy pathCombinationStrategy) {
		this(inputNeurons, outputNeurons, parallelComponentChainsBatch, directedComponentFactory.createOneToManyDirectedComponent(() -> parallelComponentChainsBatch.getComponents().size()), 
				directedComponentFactory.createManyToOneDirectedComponent(outputNeurons, pathCombinationStrategy), pathCombinationStrategy);
		this.oneToManyDirectedComponent = directedComponentFactory.createOneToManyDirectedComponent(() -> parallelComponentChainsBatch.getComponents().size());
		this.manyToOneDirectedComponent = directedComponentFactory.createManyToOneDirectedComponent(outputNeurons, pathCombinationStrategy);
		this.pathCombinationStrategy = pathCombinationStrategy;
	}
	
	/**
	 * 
	 * @param directedComponentFactory A DirectedComponentFactory instance, used to construct the nested OneToManyDirectedComponent and ManyToOneDirectedComponent.
	 * @param inputNeurons The input neurons of this graph.
	 * @param outputNeurons The output neurons of this graph.
	 * @param parallelComponentChainsBatch The batch of parallel edges within this graph, connecting
	 * @param pathCombinationStrategy The strategy specifying how the outputs of the parallel edges should be combined to
	 * produce the output activations.
	 */
	public DefaultDirectedComponentChainBipoleGraphImpl(Neurons inputNeurons, Neurons outputNeurons, 
			DefaultDirectedComponentChainBatch parallelComponentChainsBatch, OneToManyDirectedComponent<?> oneToManyDirectedComponent,
			ManyToOneDirectedComponent<?> manyToOneDirectedComponent, PathCombinationStrategy pathCombinationStrategy) {
		super(inputNeurons, outputNeurons, parallelComponentChainsBatch);
		this.oneToManyDirectedComponent = oneToManyDirectedComponent;
		this.manyToOneDirectedComponent = manyToOneDirectedComponent;
		this.pathCombinationStrategy = pathCombinationStrategy;
	}

	@Override
	public DefaultDirectedComponentChainBatch getEdges() {
		return parallelComponentChainsBatch;
	}
	
	@Override
	public DefaultDirectedComponentChainBipoleGraphActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		
		boolean originalInputIsImmutable = neuronsActivation.isImmutable();
		
		if (neuronsActivation.getFeatureCount() != getInputNeurons().getNeuronCountExcludingBias()) {
			throw new IllegalStateException(neuronsActivation.getFeatureCount() + ":" + getInputNeurons().getNeuronCountExcludingBias());
		}
		LOGGER.debug("Forward propagating through DefaultDirectedComponentChainBipoleGraphImpl");
		
		if (parallelComponentChainsBatch.getComponents().size() == 1) {
			DefaultDirectedComponentChainBatchActivation parallelChainsActivation = parallelComponentChainsBatch.forwardPropagate(Arrays.asList(neuronsActivation), context);
			return new DefaultDirectedComponentChainBipoleGraphActivationImpl(this, parallelChainsActivation, parallelChainsActivation.getOutput().get(0), originalInputIsImmutable);

		} else {
	
			OneToManyDirectedComponentActivation oneToManyDirectedComponentActivation = oneToManyDirectedComponent.forwardPropagate(neuronsActivation, context);
			
			DefaultDirectedComponentChainBatchActivation parallelChainsActivation = parallelComponentChainsBatch.forwardPropagate(oneToManyDirectedComponentActivation.getOutput(), context);
			
			ManyToOneDirectedComponentActivation manyToOneDirectedComponentActivation = manyToOneDirectedComponent.forwardPropagate(parallelChainsActivation.getOutput(), context);
			if (manyToOneDirectedComponentActivation.getOutput().getFeatureCount() != getOutputNeurons().getNeuronCountExcludingBias()) {
				throw new IllegalArgumentException("Many to one activation feature count of:" + manyToOneDirectedComponentActivation.getOutput().getFeatureCount() + " does not match output neuron count of:" + getOutputNeurons().getNeuronCountExcludingBias() );
			}
			return new DefaultDirectedComponentChainBipoleGraphActivationImpl(this, oneToManyDirectedComponentActivation, parallelChainsActivation, manyToOneDirectedComponentActivation, originalInputIsImmutable);
			
		}
	}

	@Override
	public List<DefaultChainableDirectedComponent<?, ?>> decompose() {
		return Arrays.asList(this);
	}

	@Override
	public DefaultDirectedComponentChainBipoleGraph dup() {
		return new DefaultDirectedComponentChainBipoleGraphImpl(inputNeurons, outputNeurons, parallelComponentChainsBatch.dup(),
				oneToManyDirectedComponent.dup(), manyToOneDirectedComponent.dup(),pathCombinationStrategy); 
	}
	
	@Override
	public List<NeuronsActivationFeatureOrientation> supports() {
		return NeuronsActivationFeatureOrientation.intersectLists(Arrays.asList(oneToManyDirectedComponent.supports(), parallelComponentChainsBatch.supports(), manyToOneDirectedComponent.supports()));
	}

	@Override
	public Optional<NeuronsActivationFeatureOrientation> optimisedFor() {
		return NeuronsActivationFeatureOrientation.intersectOptionals(Arrays.asList(oneToManyDirectedComponent.optimisedFor(), parallelComponentChainsBatch.optimisedFor(), manyToOneDirectedComponent.optimisedFor()));
	}

}
