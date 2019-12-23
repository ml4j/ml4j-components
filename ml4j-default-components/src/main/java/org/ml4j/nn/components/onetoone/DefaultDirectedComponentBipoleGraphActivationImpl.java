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

import org.ml4j.nn.components.DirectedComponentBatchActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.manytoone.ManyToOneDirectedComponentActivation;
import org.ml4j.nn.components.onetomany.OneToManyDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraph;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentBipoleGraphActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentBipoleGraphActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

public class DefaultDirectedComponentBipoleGraphActivationImpl extends DefaultDirectedComponentBipoleGraphActivationBase
		implements DefaultDirectedComponentBipoleGraphActivation {
	
	protected OneToManyDirectedComponentActivation inputLinkActivation;
	protected ManyToOneDirectedComponentActivation outputLinkActivation;
	protected DirectedComponentBatchActivation<NeuronsActivation, ?> edgesActivation;
	
	public DefaultDirectedComponentBipoleGraphActivationImpl(DefaultDirectedComponentBipoleGraph bipoleGraph, OneToManyDirectedComponentActivation inputLinkActivation, DirectedComponentBatchActivation<NeuronsActivation, ?> edgesActivation, 
			ManyToOneDirectedComponentActivation outputLinkActivation) {
		super(bipoleGraph, outputLinkActivation.getOutput());
		this.outputLinkActivation = outputLinkActivation;
		this.edgesActivation = edgesActivation;
		this.inputLinkActivation = inputLinkActivation;
	}
	
	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		DirectedComponentGradient<List<NeuronsActivation>> manyToOneActivation = outputLinkActivation.backPropagate(outerGradient);
		DirectedComponentGradient<List<NeuronsActivation>> edgesGradients = edgesActivation.backPropagate(manyToOneActivation);
		return inputLinkActivation.backPropagate(edgesGradients);
	}

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return Arrays.asList(this);
	}

}
