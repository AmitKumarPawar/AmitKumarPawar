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
pinMode(B1,INPUT);
pinMode(B2,INPUT);
pinMode(B3,INPUT);
pinMode(B4,INPUT);
//pinMode(A0,INPUT);  //Sets the Analog Pins A0 and A1 as INPUT Pins
//pinMode(A1,INPUT);
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
  int D1 = digitalRead(B1);
  int D2 = digitalRead(B2);
  int D3 = digitalRead(B3);
  int D4 = digitalRead(B4);

  if((D1 == 1) && (D2 == 0) && (D3 == 1) && (D4 == 0)){
    moveRobot("Forward");
  }
  
  if((D1 == 0) && (D2 == 0) && (D3 == 0) && (D4 == 0)){
    moveRobot("Stop");
  }
  
  if((D1 == 0) && (D2 == 1) && (D3 == 1) && (D4 == 0)){
    moveRobot("Right");
  }
  
  if((D1 == 1) && (D2 == 0) && (D3 == 0) && (D4 == 1)){
    moveRobot("Left");
  }
}
