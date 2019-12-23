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
package org.ml4j.nn.components.manytoone;

import java.util.List;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.manytoone.base.ManyToOneDirectedComponentBase;
import org.ml4j.nn.neurons.ImageNeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.NeuronsActivationImpl;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DummyManyToOneDirectedComponent extends ManyToOneDirectedComponentBase<DummyManyToOneDirectedComponentActivation> implements ManyToOneDirectedComponent<DummyManyToOneDirectedComponentActivation> {

	public DummyManyToOneDirectedComponent(PathCombinationStrategy pathCombinationStrategy) {
		super(pathCombinationStrategy);
	}

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyManyToOneDirectedComponent.class);
	
	/**s
	 * Serialization id.
	 */
	private static final long serialVersionUID = -7049642040068320620L;

	@Override
	public DummyManyToOneDirectedComponentActivation forwardPropagate(List<NeuronsActivation> neuronActivations,
			DirectedComponentsContext context) {
		LOGGER.debug("Mock combining multiple neurons activations into a single output neurons activation") ;
		
		if (pathCombinationStrategy == PathCombinationStrategy.FILTER_CONCAT) {
			
			NeuronsActivation firstActivation = neuronActivations.get(0).dup();
			List<NeuronsActivation> remainingActivations = neuronActivations.subList(1, neuronActivations.size());
			if (firstActivation instanceof ImageNeuronsActivation) {
				firstActivation = new NeuronsActivationImpl(firstActivation.getActivations(context.getMatrixFactory()), firstActivation.getFeatureOrientation());
			}
			NeuronsActivation result = firstActivation;

			remainingActivations.stream().forEach(a -> result.combineFeaturesInline(a, context.getMatrixFactory()));
		
			return new DummyManyToOneDirectedComponentActivation(neuronActivations.size(), result);
		
		} else {
			throw new UnsupportedOperationException();
		}
	}

	@Override
	public ManyToOneDirectedComponent<DummyManyToOneDirectedComponentActivation> dup() {
		return new DummyManyToOneDirectedComponent(pathCombinationStrategy);
	}
}
