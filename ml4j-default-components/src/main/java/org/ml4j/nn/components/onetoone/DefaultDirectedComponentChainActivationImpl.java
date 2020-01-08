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
import java.util.Collections;
import java.util.List;
import java.util.function.Supplier;
import java.util.stream.Collectors;

import org.ml4j.nn.axons.AxonsGradient;
import org.ml4j.nn.components.ChainableDirectedComponentActivation;
import org.ml4j.nn.components.DirectedComponentGradient;
import org.ml4j.nn.components.DirectedComponentGradientImpl;
import org.ml4j.nn.components.onetone.DefaultChainableDirectedComponentActivation;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChain;
import org.ml4j.nn.components.onetone.DefaultDirectedComponentChainActivation;
import org.ml4j.nn.components.onetoone.base.DefaultDirectedComponentChainActivationBase;
import org.ml4j.nn.neurons.NeuronsActivation;

/**
 * Default implementation of DefaultDirectedComponentChainActivation,  provides logic to back propagate gradient through the activation.
 * 
 * Encapsulates the activations from a forward propagation through a DefaultDirectedComponentChain.
 * 
 * @author Michael Lavelle
 */
public class DefaultDirectedComponentChainActivationImpl extends DefaultDirectedComponentChainActivationBase<DefaultDirectedComponentChain> implements DefaultDirectedComponentChainActivation {
	
	public DefaultDirectedComponentChainActivationImpl(DefaultDirectedComponentChain componentChain, List<DefaultChainableDirectedComponentActivation> activations) {
		super(componentChain, activations, activations.get(activations.size() - 1).getOutput());
	}

	@Override
	public DirectedComponentGradient<NeuronsActivation> backPropagate(
			DirectedComponentGradient<NeuronsActivation> outerGradient) {
		List<DefaultChainableDirectedComponentActivation> reversedSynapseActivations =
		        new ArrayList<>();
		    reversedSynapseActivations.addAll(getActivations());
		    Collections.reverse(reversedSynapseActivations);
		    return backPropagateAndAddToSynapseGradientList(outerGradient,
		        reversedSynapseActivations);
	}
	
	private DirectedComponentGradient<NeuronsActivation> backPropagateAndAddToSynapseGradientList(
		      DirectedComponentGradient<NeuronsActivation> outerSynapsesGradient,
		      List<DefaultChainableDirectedComponentActivation> activationsToBackPropagateThrough) {

			List<Supplier<AxonsGradient>> totalTrainableAxonsGradients = new ArrayList<>();
			totalTrainableAxonsGradients.addAll(outerSynapsesGradient.getTotalTrainableAxonsGradients());
			
		    DirectedComponentGradient<NeuronsActivation> finalGrad = outerSynapsesGradient;
		    DirectedComponentGradient<NeuronsActivation> synapsesGradient = outerSynapsesGradient;
		    List<Supplier<AxonsGradient>> finalTotalTrainableAxonsGradients = outerSynapsesGradient.getTotalTrainableAxonsGradients();
		    List<DirectedComponentGradient<NeuronsActivation>> componentGradients = new ArrayList<>();
		    for (ChainableDirectedComponentActivation<NeuronsActivation> synapsesActivation : activationsToBackPropagateThrough) {
		     
		      componentGradients.add(synapsesGradient);
		      synapsesGradient = 
		          synapsesActivation.backPropagate(synapsesGradient);
		   
		      finalTotalTrainableAxonsGradients = synapsesGradient.getTotalTrainableAxonsGradients();
		      finalGrad = synapsesGradient;
		    }
		    for (DirectedComponentGradient<NeuronsActivation> grad : componentGradients) {
		    	if (grad != finalGrad && !grad.getOutput().isImmutable()) {
		    		grad.getOutput().close();
		    	}
		    }
		    
		    
		    return new DirectedComponentGradientImpl<>(finalTotalTrainableAxonsGradients, finalGrad.getOutput());
		  }

	@Override
	public List<DefaultChainableDirectedComponentActivation> decompose() {
		return activations.stream().flatMap(a -> a.decompose().stream()).collect(Collectors.toList());
	}

}
