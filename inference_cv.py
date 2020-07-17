#!/usr/bin/env python3

from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.plugin = IECore()
        self.exec_network = None
        self.net = None
        self.input_layer = None

    def load_model(self, xml_path, num_requests, device_name, cpu_extension):
        ### TODO: Load the model ###
        self.net = IENetwork(model=xml_path, weights=xml_path.split('.')[0]+'.bin')
        ### TODO: Check for supported layers ###
        supported_layers = self.plugin.query_network(network = self.net, device_name=device_name)
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        
        
        if len(unsupported_layers)!=0:
            print("unsupported layers found:{}".format(unsupported_layers))
            if not cpu_extension==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(cpu_extension, device_name)
                supported_layers = self.plugin.query_network(network = self.net, device_name=device_name)
                unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Give the path of cpu extension")
                exit(1)
        ### TODO: Add any necessary extensions ###
       
        ### TODO: Return the loaded inference plugin ###
        self.exec_network = self.plugin.load_network(network=self.net, num_requests=num_requests, device_name=device_name)
        ### Note: You may need to update the function parameters. ###
        self.input_layer = next(iter(self.net.inputs))
        return

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        
        return self.net.inputs[self.input_layer].shape

    def exec_net(self, frame, req_id):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.exec_network.start_async(request_id=req_id, inputs={self.input_layer:frame})
        return

    def wait(self, req_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        return self.exec_network.requests[req_id].wait(-1)

    def get_output(self, req_id):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        out = self.exec_network.requests[req_id].outputs
        return [out[i] for i in out.keys()]

    
    
