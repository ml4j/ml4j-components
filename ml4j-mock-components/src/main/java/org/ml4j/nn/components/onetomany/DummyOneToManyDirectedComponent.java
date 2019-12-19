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

import java.util.function.IntSupplier;

import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetomany.base.OneToManyDirectedComponentBase;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Mock implementation of a  OneToManyDirectedComponent - a directed component which takes a single NeuronsActivation instance as input 
 * and map to many NeuronsActivation instances as output.
 * 
 * Used within component graphs where the flow through the NeuralNetwork is split into paths, eg. for skip-connections in ResNets or inception modules.
 * 
 * @author Michael Lavelle
 *
 * @param <A> The type of activation produced by this component on forward-propagation.
 */
public class DummyOneToManyDirectedComponent extends OneToManyDirectedComponentBase<DummyOneToManyDirectedComponentActivation> implements OneToManyDirectedComponent<DummyOneToManyDirectedComponentActivation> {

	private static final Logger LOGGER = LoggerFactory.getLogger(DummyOneToManyDirectedComponent.class);
	
	/**
	 * Default serialization id.
	 */
	private static final long serialVersionUID = 1L;
	
	private IntSupplier targetComponentsCountSupplier;
	
	/**
	 * DummyOneToManyDirectedComponent constructors
	 * 
	 * @param targetComponentsCountSupplier Supplier of the count of target components on the RHS of this one to many component.  This is a dynamic count to 
	 * allow this component to be constructed without knowing the number of target outputs at time of construction.
	 */
	public DummyOneToManyDirectedComponent(IntSupplier targetComponentsCountSupplier) {
		this.targetComponentsCountSupplier = targetComponentsCountSupplier;
	}

	@Override
	public DummyOneToManyDirectedComponentActivation forwardPropagate(NeuronsActivation neuronsActivation,
			DirectedComponentsContext context) {
		LOGGER.debug("Mock splitting input neurons activation into multiple output neurons activations" );
		return new DummyOneToManyDirectedComponentActivation(neuronsActivation, targetComponentsCountSupplier.getAsInt());
	}

	@Override
	public DummyOneToManyDirectedComponent dup() {
		int targetComponentsCountAtTimeOfDuplication = targetComponentsCountSupplier.getAsInt();
		return new DummyOneToManyDirectedComponent(() -> targetComponentsCountAtTimeOfDuplication);
	}
}
