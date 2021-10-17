#define B1 4
#define B2 5
#define B3 6
#define B4 7

void setup() {
  // put your setup code here, to run once:
Serial.begin(9600); // Begins the Serial Communication with the Baud rate 9600.
pinMode(10,OUTPUT); // Sets the Pins 10, 11, 12, 13 as OUTPUT Pins.
pinMode(11,OUTPUT);
pinMode(12,OUTPUT);
pinMode(13,OUTPUT);
pinMode(B1,OUTPUT);
pinMode(B2,OUTPUT);
pinMode(B3,OUTPUT);
pinMode(B4,OUTPUT);
pinMode(A0,INPUT);  //Sets the Analog Pins A0 and A1 as INPUT Pins
pinMode(A1,INPUT);
}

/* A function is a way for programmers to reuse code without having to rewrite the code.
 *  Syntax : returntype functionName(arguments){
 *            // function body
 *            return returntype;
 *            }
 */

void moveRobot(String motion){

  if(motion == "Forward"){  // RW - Fwd(10 - Pos, 11 - Neg); LW - Fwd(12 - Pos, 13 - Neg)
    digitalWrite(10,HIGH);
    digitalWrite(11,LOW);
    digitalWrite(12,HIGH);
    digitalWrite(13,LOW);
    Serial.println("Forward");
  }

  if(motion == "Backward"){  // RW - Bck(10 - Neg, 11 - Pos); LW - Bck(12 - Neg, 13 - Pos)
    digitalWrite(10,LOW);
    digitalWrite(11,HIGH);
    digitalWrite(12,LOW);
    digitalWrite(13,HIGH);
    Serial.println("Backward");
  }

  if(motion == "Left"){  // RW - Fwd(10 - Pos, 11 - Neg); LW - Bck(12 - Neg, 13 - Pos)
    digitalWrite(10,HIGH);
    digitalWrite(11,LOW);
    digitalWrite(12,LOW);
    digitalWrite(13,HIGH);
    Serial.println("Left");
  }

  if(motion == "Right"){  // RW - Bck(10 - Neg, 11 - Pos); LW - Fwd(12 - Pos, 13 - Neg)
    digitalWrite(10,LOW);
    digitalWrite(11,HIGH);
    digitalWrite(12,HIGH);
    digitalWrite(13,LOW);
    Serial.println("Right");
  }

  if(motion == "Stop"){  // RW - Stop(10 - Neg, 11 - Neg); LW - Stop(12 - Neg, 13 - Neg)
    digitalWrite(10,LOW);
    digitalWrite(11,LOW);
    digitalWrite(12,LOW);
    digitalWrite(13,LOW);
    Serial.println("Stop");
  }

}

void loop() {
  // put your main code here, to run repeatedly:
  int Right = analogRead(A0); //Reads the Analog Value on Pin A0 and Stores it in "Right"
  int Left = analogRead(A1); //Reads the Analog Value on Pin A1 and Stores it in "Left"
//  Serial.print("Value of Right Sensor is :" +String(Right));
//  Serial.print("\t");
//  Serial.println("Value of Left Sensor is :" +String(Left));
//  delay(1000);

  if((Right > 600) && (Left > 600)){ //Both Sensors Detect White Surface.
    moveRobot("Forward");
    digitalWrite(4,HIGH);
    digitalWrite(5,LOW);
    digitalWrite(6,HIGH);
    digitalWrite(7,LOW);
  }
   if((Right < 600) && (Left < 600)){ //Both Sensors Detect Black Line.
    moveRobot("Stop");
    digitalWrite(4,LOW);
    digitalWrite(5,LOW);
    digitalWrite(6,LOW);
    digitalWrite(7,LOW);
  }
   if((Right < 600) && (Left > 600)){ // Right Sensor Detects Black Line and Left Sensor Detects the White Surface
    moveRobot("Right");
    digitalWrite(4,LOW);
    digitalWrite(5,HIGH);
    digitalWrite(6,HIGH);
    digitalWrite(7,LOW);
  }
  if((Right > 600) && (Left < 600)){ // Right Sensor Detects White Surface and Left Sensor Detects the Black Line
    moveRobot("Left");
    digitalWrite(4,HIGH);
    digitalWrite(5,LOW);
    digitalWrite(6,LOW);
    digitalWrite(7,HIGH);
  }
}
