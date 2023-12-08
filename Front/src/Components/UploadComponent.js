import { ButtonGroup, TextField } from '@mui/material';
import { Button } from 'flowbite-react';
import React, { useEffect, useRef, useState } from 'react';

const UploadComponent = () => {
    const cloudinaryRef = useRef();
    const widgetRef = useRef();
    useEffect(() => {
      cloudinaryRef.current = window.cloudinary
     widgetRef.current= cloudinaryRef.current.createUploadWidget({
        cloudName:"dirbibxzp",
        uploadPreset:"hsolam4w",

      },function(error,result){
        if(!error && result && result.event === "success"){
            console.log("done! here is the image info: ",result.info.url)
            setField_info((prev) => {
                return {
                    ...prev,
                    imageAvaulable: true,
                    image: result.info.url,
                };

        })
      }})
      },[]
      )
    const [field_info, setField_info] = useState({
        inputAvailable:false,
        input:'',
        imageAvaulable:false,
        image:'',
        textFileAvailable:false,
        textFile:'',
        pdfAvailable:false,
        pdf:'',
        audioAvailable:false,
        audio:'',
        transcriptAvailable:false,
        transcript:'',
        ocrAvailable:false,
        ocr:'',
        pdfOcrAvailable:false,
        pdfOcr:'',
        textDataOcrAvailable:false,
        textDataOcr:'',

    })
    const handleChange = (e) => {
        const { name, value } = e.target;
        setField_info((prev) => {
            return {
                ...prev,
                inputAvailable: true,
                input: value,
            };
        });
      };
      const handleSubmit = async (event) => {
        
     event.preventDefault();
     if (field_info.inputAvailable) {
       
       console.log(field_info);
        
     }
    
        // try {
        //   const response = await axios.post(
        //     "https://iut-backend.onrender.com/api/registrations",
        //     submitForm,
        //     {
        //       headers: {
        //         "Content-Type": "application/json",
        //       },
        //     }
        //   );
    
        //   notify();
        // } catch (error) {
        //   console.error("Error posting data:", error);
        //   // Handle error, show an error message, etc.
        // } // 👈️ prevent page refresh
    
      };
      
  // upload image : 



    return (
        <div>
            this is upload component
            <div>
                {
                    field_info.imageAvaulable && <img src={field_info.image} alt="" />
                }
            <TextField  onChange={handleChange} id="standard-basic" label="input" variant="standard" />
            <h1>{field_info.input}</h1>
            </div>
            <ButtonGroup className='mt-6' variant="contained" aria-label="outlined primary button group">
  <Button onClick={handleSubmit}>submit</Button>
  <Button onClick={()=>widgetRef.current.open()}>iamge</Button>
  <Button>audio</Button>
  <Button>audio_file</Button>
  <Button>pdf_file</Button>
  <Button>Text_file</Button>
 
</ButtonGroup>
        </div>
    );
};

export default UploadComponent;