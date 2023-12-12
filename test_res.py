import json
import os 


with open(os.path.join('./', 'gpt4tools_val_seen_tools.json')) as f:
    gpt4tools_val_seen_tools = json.load(f)
    f.close()

for idx, item in enumerate(gpt4tools_val_seen_tools):
    for k,v in item.items():
        print('k',idx, k)
        print('v',idx, v)
    
    print()
    
    if idx > 10:
        break

with open(os.path.join('./', 'result_0.json')) as f:
    result_0 = json.load(f)
    f.close()

for item in result_0:
    for k,v in item.items():
        print(k)
    
    assert False

'''

{
    "instruction": 
        "GPT4Tools can handle various text and visual tasks, such as answering questions and providing in-depth explanations and discussions. It generates human-like text and uses tools to indirectly understand images. When referring to images, GPT4Tools follows strict file name rules. To complete visual tasks, GPT4Tools uses tools and stays loyal to observation outputs. Users can provide new images to GPT4Tools with a description, but tools must be used for subsequent tasks.\n
        
        TOOLS:\n------\n\n
        GPT4Tools has access to the following tools:\n\n> 
        Sketch Detection On Image: useful when you want to generate a scribble of the image. like: generate a scribble of this image, or generate a sketch from this image, detect the sketch from this image. The input to this tool should be a string, representing the image_path\n> 
        Predict Depth On Image: useful when you want to detect depth of the image. like: generate the depth from this image, or detect the depth map on this image, or predict the depth for this image. The input to this tool should be a string, representing the image_path\n> 
        Segment the Image: useful when you want to segment all the part of the image, but not segment a certain object.like: segment all the object in this image, or generate segmentations on this image, or segment the image,or perform segmentation on this image, or segment all the object in this image.The input to this tool should be a string, representing the image_path\n> 
        Generate Image Condition On Depth: useful when you want to generate a new real image from both the user description and depth image. like: generate a real image of a object or something from this depth image, or generate a new real image of a object or something from the depth map. The input to this tool should be a comma separated string of two, representing the image_path and the user description\n> 
        Generate Image Condition On Pose Image: useful when you want to generate a new real image from both the user description and a human pose image. like: generate a real image of a human from this human pose image, or generate a new real image of a human from this pose. The input to this tool should be a comma separated string of two, representing the image_path and the user description\n\n
        
        To use a tool, please use the following format:\n\n```\n
        Thought: Do I need to use a tool? Yes\n
        Action: the action to take, should be one of [Sketch Detection On Image, Predict Depth On Image, Segment the Image, Generate Image Condition On Depth, Generate Image Condition On Pose Image]\n
        Action Input: the input to the action\n
        Observation: the result of the action\n```\n\n

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\nThought: Do I need to use a tool? No\nAI: [your response here]\n```\n\n
        Follow file name rules and do not fake non-existent file names. Remember to provide the image file name loyally from the last tool observation.\n\nPrevious conversation:\n\n\n\n
        New input: Edit the sentence by changing the grammatical categories (e.g., change a noun to a verb). The players cheered loudly.\nGPT4Tools needs to use tools to observe images, not directly imagine them. Thoughts and observations in the conversation are only visible to GPT4Tools. When answering human questions, repeat important information. Let's think step by step.\n", 
    "input": "", 
    "output": "Thought: Do I need to use a tool? No\nAI: Loud cheering came from the players.\n", 
    "Action": [], 
    "Action Input": [], 
    "user_input": "Edit the sentence by changing the grammatical categories (e.g., change a noun to a verb). The players cheered loudly.", 
    "caption": "", 
    "filename": "", 
    "id": 29, 
    "finished_previously": false
}

{
    "instruction": 
        "GPT4Tools can handle various text and visual tasks, such as answering questions and providing in-depth explanations and discussions. It generates human-like text and uses tools to indirectly understand images. When referring to images, GPT4Tools follows strict file name rules. To complete visual tasks, GPT4Tools uses tools and stays loyal to observation outputs. Users can provide new images to GPT4Tools with a description, but tools must be used for subsequent tasks.\n %
        
        TOOLS:\n------\n\nGPT4Tools has access to the following tools:\n\n>  %
        Pose Detection On Image: useful when you want to detect the human pose of the image. like: generate human poses of this image, or generate a pose image from this image. The input to this tool should be a string, representing the image_path\n> 
        Generate Image Condition On Pose Image: useful when you want to generate a new real image from both the user description and a human pose image. like: generate a real image of a human from this human pose image, or generate a new real image of a human from this pose. The input to this tool should be a comma separated string of two, representing the image_path and the user description\n\n
        
        To use a tool, please use the following format:\n\n```\nThought: Do I need to use a tool? Yes\n %
        Action: the action to take, should be one of [Pose Detection On Image, Generate Image Condition On Pose Image]\n
        Action Input: the input to the action\nObservation: the result of the action\n```\n\n %
        
        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\n %
        Thought: Do I need to use a tool? No\nAI: [your response here]\n```\n\nFollow file name rules and do not fake non-existent file names. Remember to provide the image file name loyally from the last tool observation.\n\nPrevious conversation:\n\n %
        Human: Provide an image named cache/twzemmqi.png. 
        Description: A woman in a red coat is looking at a sink. A person in a bathroom with many sinks. A man is standing at the sink looking. The woman is looking at the very low seats. The person stands at a bathroom with four sinks. Understand the image using tools.\n
        AI: Received.\n\n
        New input: Generate a real image of a low seat from the human pose image,\n $user_input
        GPT4Tools needs to use tools to observe images, not directly imagine them. Thoughts and observations in the conversation are only visible to GPT4Tools. When answering human questions, repeat important information. Let's think step by step.\n %
        Thought: Do I need to use a tool? Yes\n
        Action: Pose Detection On Image\n
        Action Input: cache/twzemmqi.png\nObservation: Result saved as examples/oqojkvrv.png\nThought: Do I need to use a tool? Yes\nAction: Generate Image Condition On Pose Image\nAction Input: examples/oqojkvrv.png, low seat\nObservation: examples/bgswilro.png\n", 
    "input": "", 
    "output": "Thought: Do I need to use a tool? No\nAI: examples/bgswilro.png\n\n", 
    "Action": [], 
    "Action Input": [], 
    "user_input": "Generate a real image of a low seat from the human pose image,", 
    "caption": "A woman in a red coat is looking at a sink. A person in a bathroom with many sinks. A man is standing at the sink looking. The woman is looking at the very low seats. The person stands at a bathroom with four sinks.", 
    "filename": "000000438993.jpg", 
    "id": 28, 
    "finished_previously": true
}





{
"instruction": 
    "GPT4Tools can handle various text and visual tasks, such as answering questions and providing in-depth explanations and discussions. It generates human-like text and uses tools to indirectly understand images. When referring to images, GPT4Tools follows strict file name rules. To complete visual tasks, GPT4Tools uses tools and stays loyal to observation outputs. Users can provide new images to GPT4Tools with a description, but tools must be used for subsequent tasks.\n
    TOOLS:\n------\n\nGPT4Tools has access to the following tools:\n\n> 
    Generate Image Condition On Canny Image: useful when you want to generate a new real image from both the user description and a canny image. like: generate a real image of a object or something from this canny image, or generate a new real image of a object or something from this edge image. The input to this tool should be a comma separated string of two, representing the image_path and the user description.\n> 
    Edge Detection On Image: useful when you want to detect the edge of the image. like: detect the edges of this image, or canny detection on image, or perform edge detection on this image, or detect the canny image of this image. The input to this tool should be a string, representing the image_path\n\n
    To use a tool, please use the following format:\n\n```\n
    Thought: Do I need to use a tool? Yes\n
    Action: the action to take, should be one of [Generate Image Condition On Canny Image, Edge Detection On Image]\n
    Action Input: the input to the action\nObservation: the result of the action\n```\n\n
    
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:\n\n```\n
    Thought: Do I need to use a tool? No\nAI: [your response here]\n```\n\nFollow file name rules and do not fake non-existent file names. Remember to provide the image file name loyally from the last tool observation.\n\nPrevious conversation:\n\n
    Human: Provide an image named images/berdtnln.png. 
    Description: A stoplight on a busy city street in the evening. A stop light in on a busy street at dusk. There are many types of lights shining including building lights, car lights, and a street light. A busy city street with lots of cars at dusk. A street in the dark with car lights and stop lights shining. Understand the image using tools.\n
    AI: Received.\n\n
    New input: Generate a real image of a busy city street with lots of cars at dusk from the edge image,\n
    GPT4Tools needs to use tools to observe images, not directly imagine them. Thoughts and observations in the conversation are only visible to GPT4Tools. When answering human questions, repeat important information. Let's think step by step.\n", 

"input": "", 
"output": "Thought: Do I need to use a tool? Yes\nAction: Edge Detection On Image\nAction Input: images/berdtnln.png\nObservation:\nimage/ayffkvlr.png\nThought: Do I need to use a tool? Yes\nAction: Generate Image Condition On Canny Image\nAction Input: image/ayffkvlr.png, busy city street with lots of cars at dusk\nObservation:\nResult saved as examples/wfydymnx.png\nThought: Do I need to use a tool? No\nAI: Result saved as examples/wfydymnx.png\n\n", 
"Action": ["Edge Detection On Image", "Generate Image Condition On Canny Image"], 
"Action Input": [["images/berdtnln.png"], ["image/ayffkvlr.png, busy city street with lots of cars at dusk"]], 
"user_input": "Generate a real image of a busy city street with lots of cars at dusk from the edge image,", 
"caption": "A stoplight on a busy city street in the evening. A stop light in on a busy street at dusk. There are many types of lights shining including building lights, car lights, and a street light. A busy city street with lots of cars at dusk. A street in the dark with car lights and stop lights shining.", 
"filename": "000000360449.jpg", 
"id": 50, 
"finished_previously": false
}
'''